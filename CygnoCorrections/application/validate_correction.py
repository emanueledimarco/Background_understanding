# USAGE:
# 1. with wildcards: python validate_friend_tree.py --sim "sim_data/sim_run*.root"  --friend "friend_data/friend_run*.root" --data "real_data/data_run*.root"
# 2. with directories: python validate_friend_tree.py --sim sim_data/ --friend friend_data/ --data real_data/

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import uproot
import torch

selection_cfg = {
    # used for the training, don't go looser than these
    "integral_min": 6000, #2000,
    "integral_max": 50000,
    "x_min": 500,
    "x_max": 2000,
    "y_min": 500,
    "y_max": 2000,
    "min_npix": 500,
    "n_hits": 200,

    # additional
    "length_min": 0,
    "length_max": 50,
    "slimness_min": 0.8,
}

from data_reading.cluster import build_clusters_from_event
from training.clusterTraining import compute_physical_scalars_from_image

def resolve_file_paths(input_path):
    """
    Risolve un input che può essere:
    - Un singolo file ROOT
    - Una directory contenente file .root
    - Un pattern con wildcard (es. 'data/run_*.root')
    """
    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "*.root")
        files = sorted(glob.glob(pattern))
    elif "*" in input_path or "?" in input_path:
        files = sorted(glob.glob(input_path))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        # Tenta comunque il glob nel caso la wildcard sia passata racchiusa tra virgolette
        files = sorted(glob.glob(input_path))

    if not files:
        raise FileNotFoundError(f"Nessun file ROOT trovato corrispondente a: {input_path}")

    print(f"Trovati {len(files)} file per l'input '{input_path}'")
    return files


def load_clusters_from_root_files(file_list, friend_file_list=None, is_friend=False, tree_name="Events", friendtree_name="Friend", image_size=64):
    """
    Legge una lista di file ROOT e restituisce le immagini 64x64 dei cluster.
    """
    all_cluster_images = []

    # Variabili dummy per build_clusters_from_event
    dummy_scalars = ["sc_integral", "sc_rms", "sc_length", "sc_width", "sc_xmean", "sc_ymean", "sc_nhits" ]
    dummy_cond = (0.0, 0.0, 0.0)

    if is_friend and (friend_file_list is None or len(friend_file_list) != len(file_list)):
        raise ValueError("Il numero di file Friend deve corrispondere esattamente al numero di file SIM nominali!")

    for idx, main_file in enumerate(file_list):
        print(f"Reading [{idx+1}/{len(file_list)}]: {main_file}")
        
        with uproot.open(main_file) as f_in:
            tree = f_in[tree_name]
            arrays = tree.arrays(library="np")

        if is_friend:
            friend_file = friend_file_list[idx]
            print(f"  └─ Opening friend tree: {friend_file}")
            with uproot.open(friend_file) as f_friend:
                friend_tree = f_friend[friendtree_name]
                friend_arrays = friend_tree.arrays(library="np")
                
                arrays["redpix_ix"] = friend_arrays["redpix_ix_corr"]
                arrays["redpix_iy"] = friend_arrays["redpix_iy_corr"]
                arrays["redpix_iz"] = friend_arrays["redpix_iz_corr"]
                arrays["sc_redpixIdx"] = friend_arrays["sc_redpixIdx_corr"]
                arrays["sc_npix"] = friend_arrays["sc_npix_corr"]

        num_events = len(arrays["nSc"])

        for iev in range(num_events):
            clusters_evt = build_clusters_from_event(
                arrays=arrays,
                iev=iev,
                scalar_vars=dummy_scalars,
                conditions=dummy_cond,
                isdata=not is_friend,
                selection_cfg=selection_cfg
            )

            for cluster in clusters_evt:
                img_2d = cluster.to_image(image_size=image_size)
                all_cluster_images.append(img_2d)

    if len(all_cluster_images) == 0:
        return np.empty((0, 1, image_size, image_size), dtype=np.float32)

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


def run_validation_plots(sim_input, friend_input, data_input, output_dir="plot/validation_friends"):
    """
    Esegue il confronto tra SIM nominale, SIM corretta (Friend) e DATI e genera i plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Risoluzione dei file di input (gestisce file singoli, directory e wildcard)
    sim_files = resolve_file_paths(sim_input)
    friend_files = resolve_file_paths(friend_input)
    data_files = resolve_file_paths(data_input)

    # 1. Caricamento Cluster
    print("\n--- Caricamento Simulazione Nominale ---")
    sim_imgs = load_clusters_from_root_files(sim_files, is_friend=False)
    
    print("\n--- Caricamento Simulazione Corretta (Friend Tree) ---")
    friend_imgs = load_clusters_from_root_files(sim_files, friend_file_list=friend_files, is_friend=True)

    print("\n--- Caricamento Dati ---")
    data_imgs = load_clusters_from_root_files(data_files, is_friend=False)

    # 2. Calcolo degli Scalari Fisici
    print("\n--- Calcolo scalari fisici sui cluster ---")
    sim_scalars = compute_scalars_in_batches(sim_imgs, device=device)
    friend_scalars = compute_scalars_in_batches(friend_imgs, device=device)
    data_scalars = compute_scalars_in_batches(data_imgs, device=device)

    # 3. Configurazione dei Plot
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

            # C. DATA TARGET REALE
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
    parser = argparse.ArgumentParser(description="Validazione del Friend Tree rispetto a SIM e DATI (supporta cartelle o wildcard)")
    parser.add_argument("--sim", type=str, required=True, help="Path file ROOT, cartella o wildcard per SIM (es. 'sim_dir/' o 'sim_run*.root')")
    parser.add_argument("--friend", type=str, required=True, help="Path file ROOT, cartella o wildcard per Friend Tree")
    parser.add_argument("--data", type=str, required=True, help="Path file ROOT, cartella o wildcard per DATI (es. 'data_dir/' o 'data_run*.root')")
    parser.add_argument("--output-dir", type=str, default="plot/validation_friends", help="Directory di output per i grafici")

    args = parser.parse_args()

    run_validation_plots(
        sim_input=args.sim,
        friend_input=args.friend,
        data_input=args.data,
        output_dir=args.output_dir
    )
