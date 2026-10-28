import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cygno as cy

# Importazione delle funzioni PyTorch/MPS da make_2dmaps_frommidas_2.py
from make_2dmaps_frommidas import get_device, get_or_create_pedestal, swift_download_midas_file

# Stile visivo stile ROOT / Physics
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2


def save_event_map_plot(
    img_event_np,
    run_num,
    event_idx,
    output_dir,
    is_vetoed=False,
    vmax=None,
    width=4096,
    total_height=6912
):
    """
    Salva il plot 2D centrato (PDF e PNG) per il singolo evento con colormap rainbow (stile ROOT prof2d).
    I bin vuoti o sotto-soglia (0 o NaN) vengono resi visivamente trasparenti/bianchi.
    """
    fig, ax = plt.subplots(figsize=(8, 12))

    # Sostituiamo i valori <= 0 con NaN per renderli trasparenti
    display_img = np.where(img_event_np > 0, img_event_np, np.nan)

    # Colormap RAINBOW in stile ROOT (stessa usata nei profili e medie 2D)
    cmap = plt.cm.rainbow.copy()
    cmap.set_bad(color='white', alpha=0.0)

    # Estremi della scala Z
    v_max_val = vmax if vmax is not None else (np.nanmax(display_img) if np.nanmax(display_img) > 0 else 100)

    im = ax.imshow(
        display_img,
        origin='lower',
        extent=[0, width, 0, total_height],
        cmap=cmap,
        aspect='equal',
        vmin=0.1,
        vmax=v_max_val
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Charge [ADC]', fontsize=11)

    title_status = "[VETOED - SPIKE]" if is_vetoed else "[GOOD EVENT]"
    title_color = "crimson" if is_vetoed else "black"

    ax.set_title(f"Event {event_idx:04d} - Run {run_num:05d} {title_status}", fontsize=12, fontweight='bold', color=title_color, pad=10)
    ax.set_xlabel("X [pixel]", fontsize=11)
    ax.set_ylabel("Y [pixel]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, f"event_{event_idx:04d}")

    # Salvataggio centrato senza bordi bianchi esterni
    plt.savefig(f"{out_base}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{out_base}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def debug_events_from_midas(
    run_num,
    pedrun,
    max_events=20,
    vmax=100.0,
    tag="LNGS",
    tmpdir="/tmp/",
    veto_events_dir="./veto_events",
    output_dir="./debug_event_maps",
    width=4096,
    height_single_cam=2304,
    num_cameras=3,
    nsigma=3.0,
    cimax=65535
):
    device = get_device()
    total_height = height_single_cam * num_cameras

    # 1. Caricamento / Calcolo Piedistallo
    print(f"--> Caricamento/Calcolo piedistallo per il pedrun {pedrun:05d} su GPU M4...")
    ped_mean, ped_std = get_or_create_pedestal(
        pedrun=pedrun,
        device=device,
        tag=tag,
        tmpdir=tmpdir,
        width=width,
        height_single_cam=height_single_cam,
        num_cameras=num_cameras
    )

    nsigma_std = nsigma * ped_std

    # 2. Controllo Veto Eventi Spike
    veto_set = set()
    veto_file = os.path.join(veto_events_dir, f"spike_events_run{run_num:05d}.npz")
    if os.path.exists(veto_file):
        veto_data = np.load(veto_file)
        veto_set = set(veto_data["veto_events"])
        print(f"  [VETO ATTIVO] Caricati {len(veto_set)} eventi da scartare da: {veto_file}")

    # 3. Download e Apertura file MIDAS
    print(f"\n---> Debug Eventi Run MIDAS: {run_num:05d}")
    try:
        mf = swift_download_midas_file(run_num, tmpdir, tag)
    except Exception as e:
        print(f"[ERRORE] Impossibile aprire il file MIDAS per il run {run_num}: {e}")
        return

    # Sottocartelle distinte per eventi buoni e scartati
    run_out_dir_good = os.path.join(output_dir, f"run_{run_num:05d}", "good")
    run_out_dir_veto = os.path.join(output_dir, f"run_{run_num:05d}", "veto")
    os.makedirs(run_out_dir_good, exist_ok=True)
    os.makedirs(run_out_dir_veto, exist_ok=True)

    event_count = 0
    saved_good = 0
    saved_veto = 0

    mf.jump_to_start()

    for mevent in mf:
        if mevent.header.is_midas_internal_event():
            continue

        if max_events is not None and saved_good >= max_events and saved_veto >= max_events:
            print(f"\n[DEBUG] Raggiunto il limite massimo di {max_events} eventi per entrambe le categorie.")
            break

        is_vetoed = event_count in veto_set

        if max_events is not None:
            if is_vetoed and saved_veto >= max_events:
                event_count += 1
                continue
            if not is_vetoed and saved_good >= max_events:
                event_count += 1
                continue

        img_event = torch.zeros((total_height, width), dtype=torch.float32, device=device)
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
                has_camera_data = True

        if has_camera_data:
            img_event_np = img_event.cpu().numpy()

            if is_vetoed:
                save_event_map_plot(
                    img_event_np=img_event_np,
                    run_num=run_num,
                    event_idx=event_count,
                    output_dir=run_out_dir_veto,
                    is_vetoed=True,
                    vmax=vmax,
                    width=width,
                    total_height=total_height
                )
                saved_veto += 1
                print(f"  > [VETO] Salvata mappa per Evento {event_count:04d} in: {run_out_dir_veto}")
            else:
                save_event_map_plot(
                    img_event_np=img_event_np,
                    run_num=run_num,
                    event_idx=event_count,
                    output_dir=run_out_dir_good,
                    is_vetoed=False,
                    vmax=vmax,
                    width=width,
                    total_height=total_height
                )
                saved_good += 1
                print(f"  > [GOOD] Salvata mappa per Evento {event_count:04d} in: {run_out_dir_good}")

            event_count += 1

    print(f"\n=================================================")
    print(f"Debug completato per il Run {run_num:05d}!")
    print(f"  > Eventi Buoni salvati ({saved_good}): {run_out_dir_good}")
    print(f"  > Eventi Veto salvati ({saved_veto}): {run_out_dir_veto}")
    print(f"=================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera mappe 2D evento per evento con palette rainbow (Good vs Veto) per debug MIDAS (GPU M4).")
    parser.add_argument("--run", type=int, required=True, help="Numero del run MIDAS di segnale")
    parser.add_argument("--pedrun", type=int, required=True, help="Numero del run MIDAS di piedistallo")
    parser.add_argument("--max-events", type=int, default=20, help="Numero massimo di eventi da salvare per categoria (default: 20, usa -1 per tutti gli eventi)")
    parser.add_argument("--vmax", type=float, default=100.0, help="Valore massimo della z-scale in ADC per i plot 2D")
    parser.add_argument("--nsigma", type=float, default=3.0, help="Soglia n*sigma per la zero suppression")
    parser.add_argument("--tag", type=str, default="LNGS")
    parser.add_argument("--tmpdir", type=str, default="/tmp/")
    parser.add_argument("--veto-events-dir", type=str, default="./veto_events", help="Directory contenente i file .npz degli eventi spike")
    parser.add_argument("--output-dir", type=str, default="./debug_event_maps", help="Directory dove salvare le mappe 2D degli eventi")

    args = parser.parse_args()

    max_evs = None if args.max_events <= 0 else args.max_events

    debug_events_from_midas(
        run_num=args.run,
        pedrun=args.pedrun,
        max_events=max_evs,
        vmax=args.vmax,
        nsigma=args.nsigma,
        tag=args.tag,
        tmpdir=args.tmpdir,
        veto_events_dir=args.veto_events_dir,
        output_dir=args.output_dir
    )
