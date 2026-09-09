# python -m application.validate_correction --sim data/sim/recosim_cu_defaultparams/merged.root --friend data/sim/recosim_cu_defaultparams/merged_friend.root --data data/runs/recodata_run4_cu/reco_run43385_3D.root --output-dir plot/validation_cu

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot
import torch

from data_reading.cluster import build_clusters_from_event
from training.clusterTraining import compute_physical_scalars_from_image

def load_clusters_from_root(root_file, friend_file=None, is_friend=False, tree_name="Events", friendtree_name="Friend", image_size=64):
    """
    Legge un file ROOT e restituisce le immagini 64x64 dei cluster.
    Se is_friend=True, carica i dati delle collezioni 'corr' e 'sc_redpixIdx_corr'.
    """
    print(f"Reading file: {root_file}")
    
    with uproot.open(root_file) as f_in:
        tree = f_in[tree_name]
        arrays = tree.arrays(library="np")

    # Se stiamo leggendo la simulazione corretta, colleghiamo il friend tree
    if is_friend:
        if friend_file is None:
            raise ValueError("Devi fornire un friend_file per la simulazione corretta!")
        print(f"Opening friend tree: {friend_file}")
        with uproot.open(friend_file) as f_friend:
            friend_tree = f_friend[friendtree_name]
            friend_arrays = friend_tree.arrays(library="np")
            
            # Sostituiamo le collezioni dei pixel con quelle corrette
            arrays["redpix_ix"] = friend_arrays["redpix_ix_corr"]
            arrays["redpix_iy"] = friend_arrays["redpix_iy_corr"]
            arrays["redpix_iz"] = friend_arrays["redpix_iz_corr"]
            arrays["sc_redpixIdx"] = friend_arrays["sc_redpixIdx_corr"]
            arrays["sc_npix"] = friend_arrays["sc_npix_corr"]

    num_events = len(arrays["nSc"])
    all_cluster_images = []

    # Variabili dummy per build_clusters_from_event
    dummy_scalars = ["sc_integral", "sc_rms", "sc_length", "sc_width", "sc_xmean", "sc_ymean", "sc_nhits" ]
    dummy_cond = (0.0, 0.0, 0.0)

    for iev in range(num_events):
        clusters_evt = build_clusters_from_event(
            arrays=arrays,
            iev=iev,
            scalar_vars=dummy_scalars,
            conditions=dummy_cond,
            isdata=not is_friend,
            selection_cfg=None
        )

        for cluster in clusters_evt:
            img_2d = cluster.to_image(image_size=image_size)
            all_cluster_images.append(img_2d)

    if len(all_cluster_images) == 0:
        return np.empty((0, 1, image_size, image_size), dtype=np.float32)

    # Converti in tensore 4D NCHW: (N, 1, 64, 64)
    images_np = np.array(all_cluster_images, dtype=np.float32)
    images_np = np.expand_dims(images_np, axis=1)
    return images_np


def compute_scalars_in_batches(images_np, batch_size=256, device="cpu"):
    """
    Calcola i 7 scalari fisici a partire dalle immagini 2D dei cluster in batch.
    """
    if len(images_np) == 0:
        return np.empty((0, 7))

    scalars_list = []
    num_samples = len(images_np)

    for i in range(0, num_samples, batch_size):
        batch = torch.tensor(images_np[i:i + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            scalars = compute_physical_scalars_from_image(batch)
            scalars_list.append(scalars.cpu().numpy())

    return np.concatenate(scalars_list, axis=0)


def run_validation_plots(sim_file, friend_file, data_file, output_dir="plot/validation_friends"):
    """
    Esegue il confronto tra SIM nominale, SIM corretta (Friend) e DATI e genera i plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Caricamento Cluster
    print("\n--- Caricamento Simulazione Nominale ---")
    sim_imgs = load_clusters_from_root(sim_file, is_friend=False)
    
    print("\n--- Caricamento Simulazione Corretta (Friend Tree) ---")
    friend_imgs = load_clusters_from_root(sim_file, friend_file=friend_file, is_friend=True)

    print("\n--- Caricamento Dati ---")
    data_imgs = load_clusters_from_root(data_file, is_friend=False)

    # 2. Calcolo degli Scalari Fisici
    print("\n--- Calcolo scalari fisici sui cluster ---")
    sim_scalars = compute_scalars_in_batches(sim_imgs, device=device)
    friend_scalars = compute_scalars_in_batches(friend_imgs, device=device)
    data_scalars = compute_scalars_in_batches(data_imgs, device=device)

    # 3. Configurazione dei Plot (identica a validation.py)
    plot_configs = [
        ("Macro_Shape", [
            ("Integral (counts)", 0),
            ("Length (pix)", 1),
            ("Width (pix)", 2)
        ]),
        ("Micro_Topology", [
            ("$n_{pix}$", 3),
            (r"Eccentricity ($\sqrt{1 - \left(\frac{w}{l}\right)^2}$)", 4),
            ("Relative peak (Max/Integral)", 5)
        ])
    ]

    print("\n--- Generazione Grafici ---")
    for fig_name, var_setup in plot_configs:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Friend Tree Validation | {fig_name}", fontsize=16)

        for col_idx, (scalar_name, var_idx) in enumerate(var_setup):
            ax = axes[col_idx]

            # Uniamo i valori per definire un binning consistente e pulito
            all_vals = np.concatenate([
                sim_scalars[:, var_idx] if len(sim_scalars) > 0 else [],
                friend_scalars[:, var_idx] if len(friend_scalars) > 0 else [],
                data_scalars[:, var_idx] if len(data_scalars) > 0 else []
            ])

            if len(all_vals) == 0:
                continue

            vmin, vmax = np.percentile(all_vals, 1.0), np.percentile(all_vals, 99.0)
            if vmin == vmax:
                vmax += 1e-5
            bins = np.linspace(vmin, vmax, 30)

            # A. SIM DI PARTENZA (Nominale)
            if len(sim_scalars) > 0:
                ax.hist(
                    sim_scalars[:, var_idx], bins=bins, alpha=0.5,
                    histtype="step", linewidth=2, density=True,
                    label="SIM Nominale", color="tab:blue"
                )

            # B. SIM CORRETTA (Friend Tree)
            if len(friend_scalars) > 0:
                ax.hist(
                    friend_scalars[:, var_idx], bins=bins, alpha=0.7,
                    histtype="step", linewidth=2, density=True,
                    label="SIM Corretta (Flow)", color="tab:orange"
                )

            # C. DATA TARGET REALE (Punti con errori)
            if len(data_scalars) > 0:
                counts, bin_edges = np.histogram(data_scalars[:, var_idx], bins=bins)
                counts_density, _ = np.histogram(data_scalars[:, var_idx], bins=bins, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

                valid_bins = counts > 0
                scaling_factor = np.divide(
                    counts_density, counts,
                    out=np.zeros_like(counts_density),
                    where=(counts > 0)
                )
                errors_density = np.sqrt(counts) * scaling_factor

                ax.errorbar(
                    bin_centers[valid_bins], counts_density[valid_bins],
                    yerr=errors_density[valid_bins], fmt="o", markersize=4,
                    color="black", capsize=2, label="Data"
                )

            ax.set_title(scalar_name, fontweight="bold", fontsize=12)
            ax.set_ylabel("Normalized Density", fontsize=11)
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.2, linestyle="--")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        for ext in ["png", "pdf"]:
            plotname = os.path.join(output_dir, f"validation_friend_{fig_name}.{ext}")
            plt.savefig(plotname, dpi=120)
            print(f"Saved: {plotname}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validazione del Friend Tree rispetto a SIM e DATI")
    parser.add_argument("--sim", type=str, required=True, help="Path del file ROOT SIM nominale")
    parser.add_argument("--friend", type=str, required=True, help="Path del file ROOT Friend Tree generato dall'inferenza")
    parser.add_argument("--data", type=str, required=True, help="Path del file ROOT dei DATI reali")
    parser.add_argument("--output-dir", type=str, default="plot/validation_friends", help="Directory di output per i grafici")

    args = parser.parse_args()

    run_validation_plots(
        sim_file=args.sim,
        friend_file=args.friend,
        data_file=args.data,
        output_dir=args.output_dir
    )
