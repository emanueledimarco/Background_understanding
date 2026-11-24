import os
import argparse
import numpy as np
import ROOT
import cygno as cy
import torch

from event_spark_filter import clean_single_event_map, save_event_spark_check


def get_device(device_pref="auto"):
    """Rileva e restituisce il device PyTorch (MPS Metal su Apple Silicon, CUDA o CPU)."""
    if device_pref == "cpu":
        print("[CPU] Uso CPU forzato da CLI")
        return torch.device("cpu")
    if device_pref == "mps":
        if torch.backends.mps.is_available():
            print("[GPU] Trovato acceleratore Apple Silicon GPU (MPS Metal)")
            return torch.device("mps")
        print("[CPU] MPS richiesto ma non disponibile, fallback su CPU")
        return torch.device("cpu")
    if device_pref == "cuda":
        if torch.cuda.is_available():
            print("[GPU] Trovato acceleratore NVIDIA CUDA")
            return torch.device("cuda")
        print("[CPU] CUDA richiesto ma non disponibile, fallback su CPU")
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        print("[GPU] Trovato acceleratore Apple Silicon GPU (MPS Metal)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("[GPU] Trovato acceleratore NVIDIA CUDA")
        return torch.device("cuda")
    print("[CPU] GPU non disponibile, uso CPU standard")
    return torch.device("cpu")


def swift_download_midas_file(run, tmpdir, tag="LNGS"):
    print(f"--> Download/apertura file MIDAS per il run: {int(run):05d}")
    return cy.open_mid(int(run), path=tmpdir, cloud=True, tag=tag, verbose=False)


def get_or_create_pedestal(
    pedrun,
    device,
    tag="LNGS",
    tmpdir="/tmp/",
    ped_dir="./pedestals",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    **kwargs,
):
    os.makedirs(ped_dir, exist_ok=True)
    ped_cache_file = os.path.join(ped_dir, f"pedestal_run{pedrun:05d}.npz")

    if os.path.exists(ped_cache_file):
        print(f"\n[PEDESTAL] Caricamento piedistallo da cache: {ped_cache_file}")
        data = np.load(ped_cache_file)
        ped_mean_t = torch.from_numpy(data["ped_mean"]).to(device=device, dtype=torch.float32)
        ped_std_t = torch.from_numpy(data["ped_std"]).to(device=device, dtype=torch.float32)
        return ped_mean_t, ped_std_t

    print(f"\n[PEDESTAL] Calcolo piedistallo dal Run {pedrun:05d} ({device})...")
    total_height = height_single_cam * num_cameras

    sum_img = torch.zeros((total_height, width), dtype=torch.float32, device=device)
    sum_sq_img = torch.zeros((total_height, width), dtype=torch.float32, device=device)
    counts = torch.zeros((total_height, width), dtype=torch.int64, device=device)

    mf = swift_download_midas_file(pedrun, tmpdir, tag)
    mf.jump_to_start()

    event_count = 0
    for mevent in mf:
        if mevent.header.is_midas_internal_event():
            continue

        for key in mevent.banks.keys():
            if key.startswith("CAM"):
                try:
                    cam_idx = int(key.replace("CAM", ""))
                except ValueError:
                    continue

                if cam_idx >= num_cameras:
                    continue

                img_arr, _, _ = cy.daq_cam2array(mevent.banks[key])
                img_t = torch.from_numpy(img_arr).to(device=device, dtype=torch.float32)

                y_start = cam_idx * height_single_cam
                y_end = y_start + height_single_cam

                sum_img[y_start:y_end, :] += img_t
                sum_sq_img[y_start:y_end, :] += img_t**2
                counts[y_start:y_end, :] += 1

        event_count += 1
        if event_count % 10 == 0:
            print(f"  > [Piedistallo] Processati {event_count} eventi...")

    counts = torch.clamp(counts, min=1)
    ped_mean = sum_img / counts
    variance = (sum_sq_img / counts) - (ped_mean**2)
    ped_std = torch.sqrt(torch.clamp(variance, min=0.0))

    ped_mean_np = ped_mean.cpu().numpy().astype(np.float32)
    ped_std_np = ped_std.cpu().numpy().astype(np.float32)
    np.savez_compressed(ped_cache_file, ped_mean=ped_mean_np, ped_std=ped_std_np)

    return ped_mean.to(dtype=torch.float32), ped_std.to(dtype=torch.float32)


def create_2dmaps_from_midas(
    run_start,
    run_end,
    pedrun,
    output_root_file="./midas_2dmaps_pedsub.root",
    tag="LNGS",
    tmpdir="/tmp/",
    ped_dir="./pedestals",
    veto_events_dir="./veto_events",
    topo_veto_file=None,
    event_spark_veto=True,
    event_spark_nsigma=25.0,
    event_spark_skirt_nsigma=15.0,
    event_spark_max_radius=5,
    event_spark_hard_cap=400.0,
    event_spark_dilate=1,
    edge_margin=100.0,
    event_spark_check_dir=None,
    event_spark_max_plots=10,
    device_pref="auto",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    nsigma=3.0,
    cimax=65535,
):
    device = get_device(device_pref)
    total_height = height_single_cam * num_cameras

    nx_bins = 256
    ny_bins = 144 * num_cameras
    bin_x = width // nx_bins
    bin_y = total_height // ny_bins

    x_edges = np.linspace(0.0, float(width), nx_bins + 1)
    y_edges = np.linspace(0.0, float(total_height), ny_bins + 1)

    topo_veto_mask = None
    if topo_veto_file is not None:
        if not os.path.exists(topo_veto_file):
            raise FileNotFoundError(f"File veto topologico non trovato: {topo_veto_file}")
        topo_veto_mask = np.load(topo_veto_file)["veto_mask"].astype(bool)
        if topo_veto_mask.shape != (nx_bins, ny_bins):
            raise ValueError(f"Shape veto_mask {topo_veto_mask.shape} incompatibile con {(nx_bins, ny_bins)}")
        print(f"[TOPO VETO] Attivo da {topo_veto_file} ({int(np.sum(topo_veto_mask))} macro-bin)")

    interior_mask = None
    if edge_margin is not None and edge_margin > 0:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_in = (x_centers >= (x_edges[0] + edge_margin)) & (x_centers <= (x_edges[-1] - edge_margin))
        y_in = (y_centers >= (y_edges[0] + edge_margin)) & (y_centers <= (y_edges[-1] - edge_margin))
        interior_mask = np.outer(x_in, y_in)

    if event_spark_veto:
        print(
            f"[EVENT SPARK FILTER] Attivo su singolo evento "
            f"(nsigma={event_spark_nsigma}, skirt={event_spark_skirt_nsigma}, "
            f"max_radius={event_spark_max_radius}, hard_cap={event_spark_hard_cap} ADC, "
            f"dilate={event_spark_dilate}, edge_margin={edge_margin} px)"
        )

    ped_mean, ped_std = get_or_create_pedestal(
        pedrun=pedrun,
        device=device,
        tag=tag,
        tmpdir=tmpdir,
        ped_dir=ped_dir,
        width=width,
        height_single_cam=height_single_cam,
        num_cameras=num_cameras,
    )

    # Buffer singolo evento riutilizzati su GPU per evitare riallocazioni per ogni evento
    ev_charge = torch.zeros((total_height, width), dtype=torch.float32, device=device)
    ev_occ = torch.zeros((total_height, width), dtype=torch.float32, device=device)

    # Accumulatori 2D rebinnati (shape: nx_bins=256, ny_bins=432)
    sum2d_total = np.zeros((nx_bins, ny_bins), dtype=np.float64)
    occ2d_total = np.zeros((nx_bins, ny_bins), dtype=np.float64)
    event_spark_veto_map = np.zeros((nx_bins, ny_bins), dtype=np.int32)

    nsigma_std = nsigma * ped_std
    saved_check_plots = 0

    for run_num in range(run_start, run_end + 1):
        print(f"\n---> Elaborazione Run Segnale: {run_num:05d}")

        veto_set = set()
        veto_file = os.path.join(veto_events_dir, f"spike_events_run{run_num:05d}.npz")
        if os.path.exists(veto_file):
            veto_data = np.load(veto_file)
            veto_set = set(veto_data["veto_events"])
            print(f"  [VETO ATTIVO] Caricati {len(veto_set)} eventi da scartare da: {veto_file}")
            print(f"  [VETO ID] List: {sorted(list(veto_set))}")

        try:
            mf = swift_download_midas_file(run_num, tmpdir, tag)
        except Exception as e:
            print(f"ERRORE file MIDAS {run_num}: {e}")
            continue

        event_count = 0
        skipped_count = 0
        dyn_cleaned_events = 0
        dyn_rejected_events = 0
        dyn_vetoed_bins_total = 0

        mf.jump_to_start()

        for mevent in mf:
            if mevent.header.is_midas_internal_event():
                continue

            # CONTROLLO VETO RIGOROSO (SUPPORTA SIA SERIAL NUMBER CHE INDICE LOOP)
            event_serial = mevent.header.serial_number
            if (event_serial in veto_set) or (event_count in veto_set):
                print(f"  [VETO MATCH!] Scartato Evento (Serial: {event_serial}, LoopIdx: {event_count})")
                skipped_count += 1
                event_count += 1
                continue

            ev_charge.zero_()
            ev_occ.zero_()
            has_cam_data = False

            for key in mevent.banks.keys():
                if key.startswith("CAM"):
                    try:
                        cam_idx = int(key.replace("CAM", ""))
                    except ValueError:
                        continue

                    if cam_idx >= num_cameras:
                        continue

                    img_arr, _, _ = cy.daq_cam2array(mevent.banks[key])

                    y_start = cam_idx * height_single_cam
                    y_end = y_start + height_single_cam

                    img_t = torch.from_numpy(img_arr).to(device=device, dtype=torch.float32)

                    ped_mean_cam = ped_mean[y_start:y_end, :]
                    nsigma_std_cam = nsigma_std[y_start:y_end, :]

                    img_sub = img_t - ped_mean_cam
                    valid_mask = (img_sub > nsigma_std_cam) & (img_t < cimax)

                    ev_charge[y_start:y_end, :] = torch.where(valid_mask, img_sub, 0.0)
                    ev_occ[y_start:y_end, :] = valid_mask.to(torch.float32)
                    has_cam_data = True

            if has_cam_data:
                # Rebin 16x16 su GPU -> (nx_bins, ny_bins) su CPU
                ev_sum_2d = (
                    ev_charge.view(ny_bins, bin_y, nx_bins, bin_x)
                    .sum(dim=(1, 3))
                    .transpose(0, 1)
                    .contiguous()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                ev_occ_2d = (
                    ev_occ.view(ny_bins, bin_y, nx_bins, bin_x)
                    .sum(dim=(1, 3))
                    .transpose(0, 1)
                    .contiguous()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                if event_spark_veto:
                    ev_sum_clean, ev_occ_clean, ev_veto_mask, rejected, info = clean_single_event_map(
                        ev_sum=ev_sum_2d,
                        ev_occ=ev_occ_2d,
                        topo_veto_mask=topo_veto_mask,
                        interior_mask=interior_mask,
                        nsigma=event_spark_nsigma,
                        skirt_nsigma=event_spark_skirt_nsigma,
                        max_radius=event_spark_max_radius,
                        hard_cap=event_spark_hard_cap,
                        dilate=event_spark_dilate,
                    )

                    if info["n_vetoed"] > 0:
                        dyn_cleaned_events += 1
                        dyn_vetoed_bins_total += info["n_vetoed"]
                        event_spark_veto_map += ev_veto_mask.astype(np.int32)

                        if event_spark_check_dir and saved_check_plots < event_spark_max_plots:
                            ev_label = f"run{run_num:05d}_ev{event_serial}"
                            save_event_spark_check(
                                event_label=ev_label,
                                ev_sum_before=ev_sum_2d,
                                ev_occ_before=ev_occ_2d,
                                veto_mask=ev_veto_mask,
                                x_edges=x_edges,
                                y_edges=y_edges,
                                threshold=info["core_thr"],
                                check_dir=event_spark_check_dir,
                            )
                            saved_check_plots += 1

                    if rejected:
                        dyn_rejected_events += 1
                        print(
                            f"  [EVENT SPARK REJECT] Run {run_num:05d} Serial {event_serial}: "
                            f"scartato intero evento ({info['n_vetoed']} macro-bin sopra soglia)"
                        )
                        event_count += 1
                        continue

                    ev_sum_2d = ev_sum_clean
                    ev_occ_2d = ev_occ_clean

                sum2d_total += ev_sum_2d
                occ2d_total += ev_occ_2d

            event_count += 1

        print(
            f"Run {run_num:05d} completato: {event_count} eventi totali | "
            f"{skipped_count} scartati da veto_events | "
            f"{dyn_cleaned_events} eventi puliti da micro-scariche ({dyn_vetoed_bins_total} macro-bin rimossi, "
            f"{dyn_rejected_events} eventi interi scartati)."
        )

    if topo_veto_mask is not None:
        sum2d_total[topo_veto_mask] = 0.0
        occ2d_total[topo_veto_mask] = 0.0

    # Salvataggio ROOT
    print("\nSalvataggio del file ROOT di output...")
    os.makedirs(os.path.dirname(os.path.abspath(output_root_file)), exist_ok=True)
    tfout = ROOT.TFile(output_root_file, "recreate")

    int2d = ROOT.TH2D("int2d_images", "Integrated Charge Map", nx_bins, 0, width, ny_bins, 0, total_height)
    occ2d = ROOT.TH2D("occ2d_images", "Pixel Occupancy Map", nx_bins, 0, width, ny_bins, 0, total_height)
    veto2d = ROOT.TH2D("event_spark_veto2d", "Event-Level Spark Veto Count", nx_bins, 0, width, ny_bins, 0, total_height)

    ix_indices, iy_indices = np.where(occ2d_total > 0)
    if len(ix_indices) > 0:
        x_coords = (ix_indices.astype(np.float64) + 0.5) * bin_x
        y_coords = (iy_indices.astype(np.float64) + 0.5) * bin_y
        weights_occ = occ2d_total[ix_indices, iy_indices].astype(np.float64)
        weights_int = sum2d_total[ix_indices, iy_indices].astype(np.float64)
        occ2d.FillN(len(x_coords), x_coords, y_coords, weights_occ)
        int2d.FillN(len(x_coords), x_coords, y_coords, weights_int)

    vx_indices, vy_indices = np.where(event_spark_veto_map > 0)
    if len(vx_indices) > 0:
        vx_coords = (vx_indices.astype(np.float64) + 0.5) * bin_x
        vy_coords = (vy_indices.astype(np.float64) + 0.5) * bin_y
        weights_veto = event_spark_veto_map[vx_indices, vy_indices].astype(np.float64)
        veto2d.FillN(len(vx_coords), vx_coords, vy_coords, weights_veto)

    prof2d = int2d.Clone("prof2d_images")
    prof2d.Divide(occ2d)

    tfout.cd()
    occ2d.Write()
    int2d.Write()
    prof2d.Write()
    veto2d.Write()
    tfout.Close()

    print(f"FILE ROOT Salvato con successo: {output_root_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mappe 2D MIDAS con accelerazione GPU M4 PyTorch/Metal e Veto Spikes")
    parser.add_argument("--run-start", type=int, required=True)
    parser.add_argument("--run-end", type=int, required=True)
    parser.add_argument("--pedrun", type=int, required=True)
    parser.add_argument("--output", type=str, default="./midas_2dmaps_pedsub.root")
    parser.add_argument("--nsigma", type=float, default=3.0)
    parser.add_argument("--tag", type=str, default="LNGS")
    parser.add_argument("--tmpdir", type=str, default="/tmp/")
    parser.add_argument("--ped-dir", type=str, default="./pedestals")
    parser.add_argument("--veto-events-dir", type=str, default="./veto_events")
    parser.add_argument("--topo-veto-file", type=str, default=None)

    parser.add_argument("--no-event-spark-veto", action="store_true", help="Disabilita la pulizia dinamica delle scariche sui singoli eventi")
    parser.add_argument("--event-spark-nsigma", type=float, default=25.0, help="Soglia core in sigma robuste sul singolo evento (default: 25.0)")
    parser.add_argument("--event-spark-skirt-nsigma", type=float, default=15.0, help="Soglia minima di coda sul singolo evento (default: 15.0)")
    parser.add_argument("--event-spark-max-radius", type=int, default=5, help="Raggio massimo di crescita in bin sul singolo evento (default: 5)")
    parser.add_argument("--event-spark-hard-cap", type=float, default=400.0, help="Soglia fissa di sicurezza in ADC/pixel sul singolo evento (default: 400.0)")
    parser.add_argument("--event-spark-dilate", type=int, default=1, help="Dilatazione in bin attorno alla scarica del singolo evento (default: 1)")
    parser.add_argument("--edge-margin", type=float, default=100.0, help="Margine esterno in pixel escluso dalla stima statistica per-evento (default: 100.0)")
    parser.add_argument("--event-spark-check-dir", type=str, default=None, help="Cartella opzionale dove salvare i plot di controllo per singolo evento")
    parser.add_argument("--event-spark-max-plots", type=int, default=10, help="Numero massimo di plot diagnostici per singolo evento da salvare (default: 10)")

    parser.add_argument("--device", type=str, choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--cleanup-midas", action="store_true")
    parser.add_argument("--max-cache-gb", type=float, default=30.0)
    parser.add_argument("--max-downloads", type=int, default=2)
    parser.add_argument("--stall-timeout", type=int, default=90)

    args = parser.parse_args()

    create_2dmaps_from_midas(
        run_start=args.run_start,
        run_end=args.run_end,
        pedrun=args.pedrun,
        output_root_file=args.output,
        tag=args.tag,
        tmpdir=args.tmpdir,
        ped_dir=args.ped_dir,
        veto_events_dir=args.veto_events_dir,
        topo_veto_file=args.topo_veto_file,
        event_spark_veto=not args.no_event_spark_veto,
        event_spark_nsigma=args.event_spark_nsigma,
        event_spark_skirt_nsigma=args.event_spark_skirt_nsigma,
        event_spark_max_radius=args.event_spark_max_radius,
        event_spark_hard_cap=args.event_spark_hard_cap,
        event_spark_dilate=args.event_spark_dilate,
        edge_margin=args.edge_margin,
        event_spark_check_dir=args.event_spark_check_dir,
        event_spark_max_plots=args.event_spark_max_plots,
        device_pref=args.device,
        nsigma=args.nsigma,
    )