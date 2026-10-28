import os
import argparse
import numpy as np
import uproot
import ROOT

def passes_selection_mask(arrays, iev, selection_cfg):
    """
    Calcola la maschera booleana dei supercluster selezionati.
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

    total_pixels_event = len(arrays["redpix_ix"][iev])

    npix = np.zeros(nSc, dtype=int)
    slices = []
    for isc in range(nSc):
        start = int(redpix_idx[isc])
        end = int(redpix_idx[isc + 1]) if isc < nSc - 1 else total_pixels_event
        npix[isc] = end - start
        slices.append((start, end))

    if selection_cfg is None:
        return np.ones(nSc, dtype=bool), slices

    with np.errstate(divide='ignore', invalid='ignore'):
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

def make_2dmaps_s3_range(
    run_start,
    run_end,
    output_file,
    selection_cfg=None,
    base_url="https://s3.cr.cnaf.infn.it:7480/cygno:cygno-analysis/RECO/Cygno04",
    tree_name="Events",
    width=4096,
    height_single_cam=2304,
    step_size="100MB"  # O un numero di eventi es. step_size=2000
):
    total_height = height_single_cam * 3

    branches_to_read = [
        "redpix_ix", "redpix_iy", "redpix_iz", "nSc", "sc_redpixIdx",
        "sc_integral", "sc_xmean", "sc_ymean", "sc_nhits", "sc_length", "sc_width", "cam"
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    tfout = ROOT.TFile(output_file, "recreate")

    int2d  = ROOT.TH2D("int2d", "Integral 2D", 256, 0, width, 144 * 3, 0, total_height)
    occ2d  = ROOT.TH2D("occ2d", "Occupancy 2D", 256, 0, width, 144 * 3, 0, total_height)

    for run_num in range(run_start, run_end + 1):
        # s3_url = f"{base_url}/reco_run{run_num:05d}_3D.root"
        # special reco w/o vignetting
        s3_url = f"{base_url}/reco_run{run_num:05d}_3D.root"
        print(f"\n---> Apertura file remoto S3: {s3_url}")

        try:
            with uproot.open(s3_url) as f_in:
                if tree_name not in f_in:
                    print(f"WARNING: Tree {tree_name} not found in run {run_num}. Skip.")
                    continue

                tree = f_in[tree_name]
                
                # INSTEAD OF tree.arrays(), USIAMO tree.iterate()
                # step_size="50MB" scarica e processa il file in blocchi ottimizzati per la rete
                for chunk_idx, arrays in enumerate(tree.iterate(branches_to_read, library="np", step_size=step_size)):
                    num_events_chunk = len(arrays["nSc"])
                    print(f"  > Processing Block {chunk_idx + 1} ({num_events_chunk} entries)...")

                    for iev in range(num_events_chunk):
                        nSc = arrays["nSc"][iev]
                        cam = arrays["cam"][iev]
                        if nSc == 0:
                            continue

                        mask, slices = passes_selection_mask(arrays, iev, selection_cfg)
                        selected_iscs = np.where(mask)[0]
                        if len(selected_iscs) == 0:
                            continue

                        evt_x_list, evt_y_list, evt_q_list = [], [], []

                        for isc in selected_iscs:
                            start, end = slices[isc]
                            
                            x_pix = arrays["redpix_ix"][iev][start:end]
                            y_pix = arrays["redpix_iy"][iev][start:end] + cam * height_single_cam
                            q_pix = arrays["redpix_iz"][iev][start:end]

                            evt_x_list.append(x_pix)
                            evt_y_list.append(y_pix)
                            evt_q_list.append(q_pix)

                        evt_x = np.concatenate(evt_x_list).astype(np.float64)
                        evt_y = np.concatenate(evt_y_list).astype(np.float64)
                        evt_q = np.concatenate(evt_q_list).astype(np.float64)
                        evt_ones = np.ones_like(evt_x, dtype=np.float64)

                        n_points = len(evt_x)
                        int2d.FillN(n_points, evt_x, evt_y, evt_q)
                        occ2d.FillN(n_points, evt_x, evt_y, evt_ones)

        except Exception as e:
            print(f"ERRORE durante la lettura del run {run_num}: {e}")
            continue

    prof2d = int2d.Clone("prof2d")
    prof2d.Divide(occ2d)

    tfout.cd()
    prof2d.Write()
    int2d.Write()
    occ2d.Write()
    tfout.Close()

    print(f"\n==========================================")
    print(f"Elaborazione completata! Mappe salvate in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mappe 2D CYGNO da S3 su un intervallo di Run.")
    parser.add_argument("--run-start", type=int, required=True, help="Numero del primo run (es. 12407)")
    parser.add_argument("--run-end", type=int, required=True, help="Numero dell'ultimo run (es. 12410)")
    parser.add_argument("--output", type=str, required=True, help="File ROOT di output per le mappe")

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
        selection_cfg=selection_cfg
    )
