import os
import time
import shutil
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import uproot
import ROOT


def passes_selection_mask(arrays, iev, selection_cfg, has_nredpix):
    """
    Calcola la maschera booleana dei supercluster selezionati usando solo branch leggeri.
    """
    nSc = arrays["nSc"][iev]
    if nSc == 0:
        return np.array([], dtype=bool), []

    sc_integral = np.asarray(arrays["sc_integral"][iev])
    sc_xmean = np.asarray(arrays["sc_xmean"][iev])
    sc_ymean = np.asarray(arrays["sc_ymean"][iev])
    sc_nhits = np.asarray(arrays["sc_nhits"][iev])
    sc_length = np.asarray(arrays["sc_length"][iev])
    sc_width = np.asarray(arrays["sc_width"][iev])
    redpix_idx = np.asarray(arrays["sc_redpixIdx"][iev])

    total_pixels_event = int(arrays["nRedpix"][iev]) if has_nredpix else len(arrays["redpix_ix"][iev])

    npix = np.zeros(nSc, dtype=int)
    slices = []
    for isc in range(nSc):
        start = int(redpix_idx[isc])
        end = int(redpix_idx[isc + 1]) if isc < nSc - 1 else total_pixels_event
        npix[isc] = end - start
        slices.append((start, end))

    if selection_cfg is None:
        return np.ones(nSc, dtype=bool), slices

    with np.errstate(divide="ignore", invalid="ignore"):
        slimness = np.where(sc_length > 0, sc_width / sc_length, 0.0)

    mask = (
        (sc_integral > selection_cfg["integral_min"]) & (sc_integral < selection_cfg["integral_max"]) &
        (sc_xmean > selection_cfg["x_min"]) & (sc_xmean < selection_cfg["x_max"]) &
        (sc_ymean > selection_cfg["y_min"]) & (sc_ymean < selection_cfg["y_max"]) &
        (npix > selection_cfg["min_npix"]) &
        (sc_nhits > selection_cfg["n_hits"]) &
        (sc_length > selection_cfg["length_min"]) & (sc_length < selection_cfg["length_max"]) &
        (slimness > selection_cfg["slimness_min"])
    )

    return mask, slices


def download_s3_to_temp(s3_url, run_num):
    """
    Scarica il file ROOT in streaming sequenziale in /tmp con retry automatici.
    """
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    t0 = time.perf_counter()
    with session.get(s3_url, stream=True, timeout=(15, 180)) as resp:
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(prefix=f"cygno_run{run_num}_", suffix=".root", delete=False)
        try:
            with open(tmp.name, "wb") as f_out:
                shutil.copyfileobj(resp.raw, f_out, length=4 * 1024 * 1024)  # buffer 4 MB
        except Exception:
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
            raise

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    speed = size_mb / elapsed if elapsed > 0 else 0.0
    print(f"  [Run {run_num}] Scaricato {size_mb:.1f} MB in {elapsed:.1f} s ({speed:.1f} MB/s)")
    return tmp.name


def process_single_run(
    run_num,
    base_url,
    tree_name,
    selection_cfg,
    width,
    height_single_cam,
    nx_bins,
    ny_bins,
    step_size,
):
    """
    Scarica in /tmp, elabora e restituisce le griglie NumPy (int_grid, occ_grid) per un singolo run.
    """
    s3_url = f"{base_url}/reco_run{run_num:05d}_3D.root"
    total_height = height_single_cam * 3
    bin_w_x = width / nx_bins
    bin_w_y = total_height / ny_bins

    int_grid = np.zeros((nx_bins, ny_bins), dtype=np.float64)
    occ_grid = np.zeros((nx_bins, ny_bins), dtype=np.float64)

    tmp_path = None
    try:
        tmp_path = download_s3_to_temp(s3_url, run_num)

        with ThreadPoolExecutor(max_workers=4) as decomp_pool:
            with uproot.open(tmp_path, decompression_executor=decomp_pool) as f_in:
                if tree_name not in f_in:
                    print(f"  [WARNING] Tree {tree_name} non trovato nel run {run_num}. Skip.")
                    return int_grid, occ_grid

                tree = f_in[tree_name]
                available_keys = set(tree.keys())
                has_nredpix = "nRedpix" in available_keys

                pass1_branches = [
                    "nSc", "sc_redpixIdx", "sc_integral", "sc_xmean", "sc_ymean",
                    "sc_nhits", "sc_length", "sc_width", "cam"
                ]
                if has_nredpix:
                    pass1_branches.append("nRedpix")
                    pass2_branches = ["redpix_ix", "redpix_iy", "redpix_iz"]
                else:
                    pass1_branches.append("redpix_ix")
                    pass2_branches = ["redpix_iy", "redpix_iz"]

                for arrays, report in tree.iterate(pass1_branches, library="np", step_size=step_size, report=True):
                    num_events_chunk = len(arrays["nSc"])
                    selected_events = []

                    for iev in range(num_events_chunk):
                        if arrays["nSc"][iev] == 0:
                            continue
                        mask, slices = passes_selection_mask(arrays, iev, selection_cfg, has_nredpix)
                        selected_iscs = np.where(mask)[0]
                        if len(selected_iscs) > 0:
                            selected_events.append((iev, selected_iscs, slices))

                    if not selected_events:
                        continue

                    # Passaggio 2: legge i pixel solo per i blocchi che hanno cluster selezionati
                    pix_arrays = tree.arrays(
                        pass2_branches,
                        entry_start=report.start,
                        entry_stop=report.stop,
                        library="np",
                    )
                    redpix_ix_chunk = pix_arrays["redpix_ix"] if has_nredpix else arrays["redpix_ix"]
                    redpix_iy_chunk = pix_arrays["redpix_iy"]
                    redpix_iz_chunk = pix_arrays["redpix_iz"]

                    chunk_x_list, chunk_y_list, chunk_q_list = [], [], []
                    for iev, selected_iscs, slices in selected_events:
                        y_offset = arrays["cam"][iev] * height_single_cam
                        ix_ev = redpix_ix_chunk[iev]
                        iy_ev = redpix_iy_chunk[iev]
                        iz_ev = redpix_iz_chunk[iev]

                        for isc in selected_iscs:
                            start, end = slices[isc]
                            chunk_x_list.append(ix_ev[start:end])
                            chunk_y_list.append(iy_ev[start:end] + y_offset)
                            chunk_q_list.append(iz_ev[start:end])

                    all_x = np.concatenate(chunk_x_list)
                    all_y = np.concatenate(chunk_y_list)
                    all_q = np.concatenate(chunk_q_list).astype(np.float64)

                    in_bounds = (all_x >= 0) & (all_x < width) & (all_y >= 0) & (all_y < total_height)
                    all_x = all_x[in_bounds]
                    all_y = all_y[in_bounds]
                    all_q = all_q[in_bounds]

                    bx = (all_x / bin_w_x).astype(np.int64)
                    by = (all_y / bin_w_y).astype(np.int64)
                    flat_idx = bx * ny_bins + by

                    int_grid += np.bincount(flat_idx, weights=all_q, minlength=nx_bins * ny_bins).reshape(nx_bins, ny_bins)
                    occ_grid += np.bincount(flat_idx, minlength=nx_bins * ny_bins).reshape(nx_bins, ny_bins)

    except Exception as e:
        print(f"  [ERRORE] Run {run_num}: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return int_grid, occ_grid


def make_2dmaps_s3_range(
    run_start,
    run_end,
    output_file,
    selection_cfg=None,
    base_url="https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Cygno04",
    tree_name="Events",
    width=4096,
    height_single_cam=2304,
    step_size="50MB",
    workers=1,
):
    nx_bins = 256
    ny_bins = 144 * 3
    total_height = height_single_cam * 3

    runs = list(range(run_start, run_end + 1))
    print(f"\n---> Elaborazione {len(runs)} run ({run_start}-{run_end}) con {workers} worker...")

    total_int = np.zeros((nx_bins, ny_bins), dtype=np.float64)
    total_occ = np.zeros((nx_bins, ny_bins), dtype=np.float64)

    if workers <= 1 or len(runs) == 1:
        for run_num in runs:
            ig, og = process_single_run(
                run_num, base_url, tree_name, selection_cfg,
                width, height_single_cam, nx_bins, ny_bins, step_size
            )
            total_int += ig
            total_occ += og
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(runs))) as executor:
            futures = [
                executor.submit(
                    process_single_run,
                    run_num, base_url, tree_name, selection_cfg,
                    width, height_single_cam, nx_bins, ny_bins, step_size
                )
                for run_num in runs
            ]
            for future in as_completed(futures):
                ig, og = future.result()
                total_int += ig
                total_occ += og

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    tfout = ROOT.TFile(output_file, "recreate")

    int2d = ROOT.TH2D("int2d", "Integral 2D", nx_bins, 0, width, ny_bins, 0, total_height)
    occ2d = ROOT.TH2D("occ2d", "Occupancy 2D", nx_bins, 0, width, ny_bins, 0, total_height)

    for ix in range(nx_bins):
        for iy in range(ny_bins):
            vo = total_occ[ix, iy]
            if vo > 0:
                int2d.SetBinContent(ix + 1, iy + 1, total_int[ix, iy])
                occ2d.SetBinContent(ix + 1, iy + 1, vo)

    total_entries = float(np.sum(total_occ))
    int2d.SetEntries(total_entries)
    occ2d.SetEntries(total_entries)

    prof2d = int2d.Clone("prof2d")
    prof2d.Divide(occ2d)

    tfout.cd()
    prof2d.Write()
    int2d.Write()
    occ2d.Write()
    tfout.Close()

    print(f"==========================================")
    print(f"Elaborazione completata! Mappe salvate in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mappe 2D CYGNO da S3 su un intervallo di Run.")
    parser.add_argument("--run-start", type=int, required=True, help="Numero del primo run (es. 12407)")
    parser.add_argument("--run-end", type=int, required=True, help="Numero dell'ultimo run (es. 12410)")
    parser.add_argument("--output", type=str, required=True, help="File ROOT di output per le mappe")
    parser.add_argument("--workers", type=int, default=1, help="Numero di worker paralleli sui run del range (default: 1)")

    args = parser.parse_args()

    selection_cfg = {
        "integral_min": 0,
        "integral_max": 1e6,
        "x_min": 0,
        "x_max": 4096,
        "y_min": 0,
        "y_max": 2304,
        "min_npix": 5,
        "n_hits": 0,
        "length_min": 0,
        "length_max": 7000,
        "slimness_min": 0.0
    }

    make_2dmaps_s3_range(
        run_start=args.run_start,
        run_end=args.run_end,
        output_file=args.output,
        selection_cfg=selection_cfg,
        workers=args.workers,
    )