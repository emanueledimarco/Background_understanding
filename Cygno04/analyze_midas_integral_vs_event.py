import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cygno as cy

from make_2dmaps_frommidas import get_device, get_or_create_pedestal, swift_download_midas_file

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2


def analyze_mean_vs_event_single_run(
    run_num,
    ped_mean,
    ped_std,
    device,
    xmin=None, xmax=None,
    ymin=None, ymax=None,
    nsigma=3.0,
    cimax=65535,
    cutoff=None,
    factor=2.0,
    tag="LNGS",
    tmpdir="/tmp/",
    output_dir="./plots_mean_vs_event",
    veto_dir="./veto_events",
    width=4096,
    height_single_cam=2304,
    num_cameras=3
):
    total_height = height_single_cam * num_cameras
    nsigma_std = nsigma * ped_std

    # Maschera ROI su GPU
    x_indices = torch.arange(width, device=device)
    y_indices = torch.arange(total_height, device=device)

    x_mask = torch.ones(width, dtype=torch.bool, device=device)
    if xmin is not None: x_mask &= (x_indices >= xmin)
    if xmax is not None: x_mask &= (x_indices <= xmax)

    y_mask = torch.ones(total_height, dtype=torch.bool, device=device)
    if ymin is not None: y_mask &= (y_indices >= ymin)
    if ymax is not None: y_mask &= (y_indices <= ymax)

    roi_mask = y_mask.unsqueeze(1) & x_mask.unsqueeze(0)

    print(f"\n---> Elaborazione Run Segnale MIDAS: {run_num:05d}")
    try:
        mf = swift_download_midas_file(run_num, tmpdir, tag)
    except Exception as e:
        print(f"[ERRORE] Impossibile aprire il file MIDAS per il run {run_num}: {e}")
        return

    event_numbers = []
    means = []

    mf.jump_to_start()

    for mevent in mf:
        if mevent.header.is_midas_internal_event():
            continue

        # Recupero del numero di serie nativo dell'evento MIDAS
        serial_num = mevent.header.serial_number

        img_event = torch.zeros((total_height, width), dtype=torch.float32, device=device)
        mask_event = torch.zeros((total_height, width), dtype=torch.bool, device=device)
        has_camera_data = False

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

                img_clean = torch.where(valid_mask, img_sub, 0.0)
                img_event[y_start:y_end, :] = img_clean
                mask_event[y_start:y_end, :] = valid_mask
                has_camera_data = True

        if has_camera_data:
            roi_valid_mask = roi_mask & mask_event
            n_pixels = torch.sum(roi_valid_mask).item()

            if n_pixels > 0:
                event_sum = torch.sum(img_event[roi_mask]).item()
                event_mean = event_sum / float(n_pixels)
            else:
                event_mean = 0.0

            # Salviamo il numero di serie invece dell'indice di loop
            event_numbers.append(serial_num)
            means.append(event_mean)

    event_numbers = np.array(event_numbers)
    means = np.array(means)

    if len(means) == 0:
        print(f"[WARNING] Nessun evento valido trovato per il run {run_num:05d}")
        return

    # ---------------------------------------------------------
    # DETERMINAZIONE DELLA SOGLIA DI VETO (FISSA O RELATIVA)
    # ---------------------------------------------------------
    avg_mean = np.mean(means)

    if cutoff is not None:
        threshold_val = cutoff
        cutoff_label = f"Fixed Cut = {cutoff:.2f} ADC/pixel"
    else:
        threshold_val = factor * avg_mean
        cutoff_label = f"{factor:.1f}$\times$Mean = {threshold_val:.2f} ADC/pixel"

    spike_mask = means > threshold_val
    spike_events = event_numbers[spike_mask]  # Array dei serial_num da vetare

    clean_means = means[~spike_mask]
    clean_avg = np.mean(clean_means) if len(clean_means) > 0 else 0

    print(f"\n[ANALISI SPIKE RUN {run_num:05d}] Trovati {len(spike_events)} eventi con carica/pixel > {threshold_val:.2f} ADC/pixel:")
    print(f"  > Eventi Spike Serial Number: {spike_events.tolist()}")
    print(f"  > Media prima del veto: {avg_mean:.2f} ADC/pixel")
    print(f"  > Media pulita dopo veto: {clean_avg:.2f} ADC/pixel")

    # Salvataggio del file .npz di veto specifico per questo run_num
    os.makedirs(veto_dir, exist_ok=True)
    veto_file_path = os.path.join(veto_dir, f"spike_events_run{run_num:05d}.npz")
    np.savez_compressed(
        veto_file_path,
        veto_events=spike_events,
        run_num=run_num,
        threshold_val=threshold_val,
        cutoff=cutoff if cutoff is not None else np.nan
    )
    print(f"  > File veto eventi salvato in: {veto_file_path}")

    # Plot
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, f"mean_vs_event_run{run_num:05d}")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(event_numbers[~spike_mask], means[~spike_mask], 'ko', markersize=3, label='Good Events')
    if len(spike_events) > 0:
        ax.plot(event_numbers[spike_mask], means[spike_mask], 'ro', markersize=5, label=f'Spikes (N={len(spike_events)})')

    ax.axhline(clean_avg, color='crimson', linestyle='--', linewidth=1.5, label=f'Clean Mean = {clean_avg:.2f} ADC/pixel')
    ax.axhline(threshold_val, color='orange', linestyle=':', linewidth=1.5, label=f'Veto Cutoff ({cutoff_label})')

    ax.set_title(f"Mean Pixel Charge vs Event Serial Number (Run {run_num:05d})", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Event Serial Number", fontsize=12)
    ax.set_ylabel("Mean ROI Charge [ADC / pixel]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.legend(loc='upper right', frameon=True)

    plt.savefig(f"{out_base}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{out_base}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def analyze_mean_vs_event_range(
    run_start,
    run_end,
    pedrun,
    xmin=None, xmax=None,
    ymin=None, ymax=None,
    nsigma=3.0,
    cutoff=None,
    factor=2.0,
    tag="LNGS",
    tmpdir="/tmp/",
    output_dir="./plots_mean_vs_event",
    veto_dir="./veto_events"
):
    device = get_device()

    print(f"--> Caricamento/Calcolo piedistallo per il pedrun {pedrun:05d} su GPU M4...")
    ped_mean, ped_std = get_or_create_pedestal(
        pedrun=pedrun,
        device=device,
        tag=tag,
        tmpdir=tmpdir
    )

    for run_num in range(run_start, run_end + 1):
        analyze_mean_vs_event_single_run(
            run_num=run_num,
            ped_mean=ped_mean,
            ped_std=ped_std,
            device=device,
            xmin=xmin, xmax=xmax,
            ymin=ymin, ymax=ymax,
            nsigma=nsigma,
            cutoff=cutoff,
            factor=factor,
            tag=tag,
            tmpdir=tmpdir,
            output_dir=output_dir,
            veto_dir=veto_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisi medie eventi MIDAS su un range di run e salvataggio veto spikes usando Serial Number.")
    parser.add_argument("--run-start", type=int, required=True, help="Run iniziale del range")
    parser.add_argument("--run-end", type=int, required=True, help="Run finale del range")
    parser.add_argument("--pedrun", type=int, required=True, help="Numero del run di piedistallo")
    parser.add_argument("--xmin", type=float, default=None, help="Taglio min X [pixel]")
    parser.add_argument("--xmax", type=float, default=None, help="Taglio max X [pixel]")
    parser.add_argument("--ymin", type=float, default=None, help="Taglio min Y [pixel]")
    parser.add_argument("--ymax", type=float, default=None, help="Taglio max Y [pixel]")
    parser.add_argument("--nsigma", type=float, default=3.0, help="Soglia zero suppression n*sigma (può essere negativa se si desidera disattivarla)")
    parser.add_argument("--cutoff", "-c", type=float, default=None, help="Soglia FISSA in ADC/pixel per scartare gli eventi spike")
    parser.add_argument("--factor", type=float, default=2.0, help="Soglia RELATIVA rispetto alla media (usata solo se --cutoff e' None)")
    parser.add_argument("--tag", type=str, default="LNGS")
    parser.add_argument("--tmpdir", type=str, default="/tmp/")
    parser.add_argument("--output-dir", type=str, default="./plots_mean_vs_event")
    parser.add_argument("--veto-dir", type=str, default="./veto_events")

    args = parser.parse_args()

    analyze_mean_vs_event_range(
        run_start=args.run_start,
        run_end=args.run_end,
        pedrun=args.pedrun,
        xmin=args.xmin, xmax=args.xmax,
        ymin=args.ymin, ymax=args.ymax,
        nsigma=args.nsigma,
        cutoff=args.cutoff,
        factor=args.factor,
        tag=args.tag,
        tmpdir=args.tmpdir,
        output_dir=args.output_dir,
        veto_dir=args.veto_dir
    )
