import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_reading.read_data_2D import _get_cached_cluster_image
from training.clusterTraining import (
    CygnoTransportModel,
    _resolve_device,
    build_dataloader,
    compute_physical_scalars_from_image,
)


def _find_matching_clusters(cluster_dict, target_key, atol=1e-4):
    target_arr = np.asarray(target_key, dtype=np.float64)
    matched = []
    for k, clusters in cluster_dict.items():
        k_arr = np.asarray(k, dtype=np.float64)
        if k_arr.shape == target_arr.shape and np.all(
            np.abs(k_arr - target_arr) < atol
        ):
            matched.extend(clusters)
    return matched


@torch.inference_mode()
def _clusters_to_scalars(clusters, device, chunk_size=1024):
    if not clusters:
        return np.empty((0, 8), dtype=np.float32)

    all_scalars = []
    for start in range(0, len(clusters), chunk_size):
        chunk = clusters[start : start + chunk_size]
        imgs = torch.stack(
            [_get_cached_cluster_image(c, 64) for c in chunk], dim=0
        ).unsqueeze(1)
        imgs = F.relu(imgs.to(device))
        scalars = compute_physical_scalars_from_image(imgs)
        all_scalars.append(scalars.cpu().numpy())

    return np.concatenate(all_scalars, axis=0)


@torch.inference_mode()
def _transport_clusters_to_scalars(
    model, sim_clusters, sim_key, data_key, device, chunk_size=1024
):
    if not sim_clusters:
        return (
            np.empty((0, 8), dtype=np.float32),
            np.empty((0, 8), dtype=np.float32),
        )

    sim_cond_base = torch.tensor(
        [sim_key], dtype=torch.float32, device=device
    )
    data_cond_base = torch.tensor(
        [data_key], dtype=torch.float32, device=device
    )

    all_sim_scalars = []
    all_pred_scalars = []

    for start in range(0, len(sim_clusters), chunk_size):
        chunk = sim_clusters[start : start + chunk_size]
        m = len(chunk)

        imgs = torch.stack(
            [_get_cached_cluster_image(c, 64) for c in chunk], dim=0
        ).unsqueeze(1)
        s_img = F.relu(imgs.to(device))
        s_cond = sim_cond_base.expand(m, -1)
        d_cond = data_cond_base.expand(m, -1)

        s_scal = compute_physical_scalars_from_image(s_img)
        out = model(s_img, s_cond, d_cond, s_scal)
        pred_img = F.relu(out["pred_images"])
        p_scal = compute_physical_scalars_from_image(pred_img)

        all_sim_scalars.append(s_scal.cpu().numpy())
        all_pred_scalars.append(p_scal.cpu().numpy())

    return (
        np.concatenate(all_sim_scalars, axis=0),
        np.concatenate(all_pred_scalars, axis=0),
    )


@torch.inference_mode()
def run_validation_sweep_from_dict(
    model,
    metadata,
    loader,
    inputfile,
    target_sim_key,
    sweep_var="H",
    device=None,
    chunk_size=1024,
):
    device = _resolve_device(device)
    model.to(device).eval()

    # -----------------------
    # 1. Estrazione delle chiavi REALI esistenti
    # -----------------------
    z_val, alpha_val, lambda_val = target_sim_key
    data_keys = metadata["keys"]["data_keys"]

    # Filtriamo solo le chiavi DATA reali che hanno lo z_val corretto
    valid_keys = [k for k in data_keys if k[0] == z_val]
    if not valid_keys:
        raise ValueError(f"Nessuna chiave DATA trovata per z = {z_val}")

    # Ordiniamo le chiavi in base alla variabile su cui vogliamo fare lo sweep
    if sweep_var == "H":
        valid_keys_sorted = sorted(valid_keys, key=lambda x: x[3])
    elif sweep_var == "P":
        valid_keys_sorted = sorted(valid_keys, key=lambda x: x[1])
    elif sweep_var == "T":
        valid_keys_sorted = sorted(valid_keys, key=lambda x: x[2])
    else:
        raise ValueError("sweep_var deve essere 'H', 'P' o 'T'")

    # Selezioniamo 3 chiavi REALI esistenti: Min, Med, Max
    idx_min = 0
    idx_med = len(valid_keys_sorted) // 2
    idx_max = len(valid_keys_sorted) - 1

    chosen_keys = [
        valid_keys_sorted[idx_min],
        valid_keys_sorted[idx_med],
        valid_keys_sorted[idx_max],
    ]

    all_data_scalars_dict = {}
    all_pred_scalars_dict = {}
    all_sim_scalars_dict = {}

    atol = 1e-4
    dataset = getattr(loader, "dataset", None)

    # -----------------------
    # 2. Accumulo Dati e Inferenza (percorso rapido diretto da dizionario in RAM)
    # -----------------------
    if (
        dataset is not None
        and hasattr(dataset, "sim_dict")
        and hasattr(dataset, "data_dict")
    ):
        sim_clusters = _find_matching_clusters(
            dataset.sim_dict, target_sim_key, atol=atol
        )
        print(
            f"\tDirect dictionary lookup: {len(sim_clusters)} SIM clusters for {target_sim_key}"
        )

        for current_key in chosen_keys:
            target_P, target_T, target_H = (
                current_key[1],
                current_key[2],
                current_key[3],
            )
            row_key = (target_P, target_T, target_H)

            data_clusters = _find_matching_clusters(
                dataset.data_dict, current_key, atol=atol
            )
            real_scalars = _clusters_to_scalars(
                data_clusters, device=device, chunk_size=chunk_size
            )
            sim_scalars, pred_scalars = _transport_clusters_to_scalars(
                model,
                sim_clusters,
                sim_key=target_sim_key,
                data_key=current_key,
                device=device,
                chunk_size=chunk_size,
            )

            all_data_scalars_dict[row_key] = (
                [real_scalars] if len(real_scalars) > 0 else []
            )
            all_sim_scalars_dict[row_key] = (
                [sim_scalars] if len(sim_scalars) > 0 else []
            )
            all_pred_scalars_dict[row_key] = (
                [pred_scalars] if len(pred_scalars) > 0 else []
            )
    else:
        # Fallback su DataLoader con filtro CPU prima del trasferimento su GPU
        for ib, batch in enumerate(loader):
            if ib % 10 == 0:
                print(f"\tProcessing cluster batch # {ib}...")

            sim_cond_cpu = batch["sim_cond"]
            data_cond_cpu = batch["data_cond"]

            s_cond_2d = (
                sim_cond_cpu[:, 0, :]
                if sim_cond_cpu.ndim == 3
                else sim_cond_cpu
            )
            d_cond_2d = (
                data_cond_cpu[:, 0, :]
                if data_cond_cpu.ndim == 3
                else data_cond_cpu
            )

            batch_data_z = d_cond_2d[:, 0].numpy()
            batch_data_P = d_cond_2d[:, 1].numpy()
            batch_data_T = d_cond_2d[:, 2].numpy()
            batch_data_H = d_cond_2d[:, 3].numpy()

            batch_sim_z = s_cond_2d[:, 0].numpy()
            batch_sim_alpha = s_cond_2d[:, 1].numpy()
            batch_sim_lambda = s_cond_2d[:, 2].numpy()

            sim_mask = (
                (np.abs(batch_sim_z - z_val) < atol)
                & (np.abs(batch_sim_alpha - alpha_val) < atol)
                & (np.abs(batch_sim_lambda - lambda_val) < atol)
            )
            has_sim = bool(np.any(sim_mask))

            if has_sim:
                torch_sim_mask = torch.from_numpy(sim_mask)
                matched_sim = F.relu(
                    batch["sim_images"][torch_sim_mask].to(device)
                )
                matched_cond = s_cond_2d[torch_sim_mask].to(device)
                B_s, N_s, H_s, W_s = matched_sim.shape
                sim_flat = matched_sim.view(B_s * N_s, 1, H_s, W_s)
                sim_cond_flat = matched_cond.repeat_interleave(N_s, dim=0)
                sim_scalars_flat = compute_physical_scalars_from_image(sim_flat)

            for current_key in chosen_keys:
                target_P, target_T, target_H = (
                    current_key[1],
                    current_key[2],
                    current_key[3],
                )
                row_key = (target_P, target_T, target_H)

                if row_key not in all_data_scalars_dict:
                    all_data_scalars_dict[row_key] = []
                    all_pred_scalars_dict[row_key] = []
                    all_sim_scalars_dict[row_key] = []

                data_mask = (
                    (np.abs(batch_data_z - z_val) < atol)
                    & (np.abs(batch_data_P - target_P) < atol)
                    & (np.abs(batch_data_T - target_T) < atol)
                    & (np.abs(batch_data_H - target_H) < atol)
                )

                if np.any(data_mask):
                    matched_data = F.relu(
                        batch["data_images"][torch.from_numpy(data_mask)].to(
                            device
                        )
                    )
                    B_d, N_d, H_d, W_d = matched_data.shape
                    data_flat = matched_data.view(B_d * N_d, 1, H_d, W_d)
                    data_scalars = compute_physical_scalars_from_image(
                        data_flat
                    )
                    all_data_scalars_dict[row_key].append(
                        data_scalars.cpu().numpy()
                    )

                if has_sim:
                    target_cond_flat = torch.tensor(
                        [[z_val, target_P, target_T, target_H]],
                        dtype=torch.float32,
                        device=device,
                    ).expand(B_s * N_s, -1)

                    out = model(
                        sim_flat,
                        sim_cond_flat,
                        target_cond_flat,
                        sim_scalars_flat,
                    )
                    pred_clamped_flat = F.relu(out["pred_images"])
                    pred_scalars = compute_physical_scalars_from_image(
                        pred_clamped_flat
                    )

                    all_pred_scalars_dict[row_key].append(
                        pred_scalars.cpu().numpy()
                    )
                    all_sim_scalars_dict[row_key].append(
                        sim_scalars_flat.cpu().numpy()
                    )

    # -----------------------
    # 3. Configurazione e generazione dei Grafici
    # -----------------------
    print("===> Done filling distributions. Now plotting...")

    output_dir = "plot/validation_plots"
    os.makedirs(output_dir, exist_ok=True)
    sim_key_str = "_".join(f"{x:g}".replace(".", "p") for x in target_sim_key)

    # Indici in compute_physical_scalars_from_image (8 scalari):
    # 0: log10(integral+1), 1: length, 2: width, 3: log10(area+1),
    # 4: x_mean, 5: y_mean, 6: eccentricity, 7: rel_peak
    plot_configs = [
        (
            "Macro_Shape",
            [
                (r"$\log_{10}(\mathrm{Integral})$", 0),
                ("Length (pix)", 1),
                ("Width (pix)", 2),
            ],
        ),
        (
            "Micro_Topology",
            [
                (r"$\log_{10}(n_{\mathrm{pix}})$", 3),
                (
                    r"Eccentricity ($\sqrt{1 - \left(\frac{w}{l}\right)^2}$)",
                    6,
                ),
                ("Relative peak (Top4/Integral)", 7),
            ],
        ),
    ]

    for fig_name, var_setup in plot_configs:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(
            f"{fig_name} | Variable {sweep_var} | Central SIM: z={z_val} cm, $\\alpha$={alpha_val}, $\\lambda$={lambda_val} mm",
            fontsize=16,
        )

        bins = {}

        for row_idx, current_key in enumerate(chosen_keys):
            target_P, target_T, target_H = (
                current_key[1],
                current_key[2],
                current_key[3],
            )
            row_key = (target_P, target_T, target_H)

            real_scalars = (
                np.concatenate(all_data_scalars_dict[row_key], axis=0)
                if all_data_scalars_dict[row_key]
                else np.empty((0, 8))
            )
            pred_scalars = (
                np.concatenate(all_pred_scalars_dict[row_key], axis=0)
                if all_pred_scalars_dict[row_key]
                else np.empty((0, 8))
            )
            sim_scalars = (
                np.concatenate(all_sim_scalars_dict[row_key], axis=0)
                if all_sim_scalars_dict[row_key]
                else np.empty((0, 8))
            )

            for col_idx, (scalar_name, var_idx) in enumerate(var_setup):
                ax = axes[row_idx, col_idx]

                non_empty = [
                    arr[:, var_idx]
                    for arr in (sim_scalars, pred_scalars, real_scalars)
                    if len(arr) > 0
                ]
                if not non_empty:
                    ax.axis("off")
                    continue

                all_vals = np.concatenate(non_empty)
                vmin, vmax = np.percentile(all_vals, 1.0), np.percentile(
                    all_vals, 99.0
                )
                if vmin == vmax:
                    vmax += 1e-5

                if col_idx not in bins:
                    bins[col_idx] = np.linspace(vmin, vmax, 30)

                # A. SIM DI PARTENZA
                if len(sim_scalars) > 0:
                    ax.hist(
                        sim_scalars[:, var_idx],
                        bins=bins[col_idx],
                        alpha=0.5,
                        histtype="step",
                        linewidth=2,
                        density=True,
                        label="Central SIM",
                        color="tab:blue",
                    )

                # B. PRED TRASPORTATI
                if len(pred_scalars) > 0:
                    ax.hist(
                        pred_scalars[:, var_idx],
                        bins=bins[col_idx],
                        alpha=0.7,
                        histtype="step",
                        linewidth=2,
                        density=True,
                        label="Corr. SIM",
                        color="tab:orange",
                    )

                # C. DATA TARGET REALE
                if len(real_scalars) > 0:
                    counts, bin_edges = np.histogram(
                        real_scalars[:, var_idx], bins=bins[col_idx]
                    )
                    counts_density, _ = np.histogram(
                        real_scalars[:, var_idx],
                        bins=bins[col_idx],
                        density=True,
                    )
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                    valid_bins = counts > 0
                    scaling_factor = np.divide(
                        counts_density,
                        counts,
                        out=np.zeros_like(counts_density),
                        where=(counts > 0),
                    )
                    errors_density = np.sqrt(counts) * scaling_factor

                    ax.errorbar(
                        bin_centers[valid_bins],
                        counts_density[valid_bins],
                        yerr=errors_density[valid_bins],
                        fmt="o",
                        markersize=4,
                        color="black",
                        capsize=2,
                        label="Data",
                    )

                if row_idx == 0:
                    ax.set_title(scalar_name, fontweight="bold", fontsize=12)
                if col_idx == 0:
                    ax.set_ylabel(
                        f"P={target_P}bar, T={target_T}C, H={target_H}ppk\n\nClusters",
                        rotation=90,
                        labelpad=10,
                        fontsize=13,
                    )

                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.2, linestyle="--")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        for ext in ["png", "pdf"]:
            plotname = os.path.join(
                output_dir,
                f"fullstat_{fig_name}_sweep_{sweep_var}_SIM_{sim_key_str}.{ext}",
            )
            plt.savefig(plotname, dpi=120)
            print(f"Saved validation plot in {plotname}")
        plt.close(fig)