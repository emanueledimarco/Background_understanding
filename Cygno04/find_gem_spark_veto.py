import os
import argparse
from collections import deque
import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def binary_dilate_2d(mask, radius=1):
    """
    Dilatazione morfologica 2D in puro NumPy (aggiunge un margine di 'radius' bin
    attorno a ciascun bin marcato come scarica).
    """
    if radius <= 0:
        return mask.copy()

    nx, ny = mask.shape
    dilated = mask.copy()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            x_src_0 = max(0, -dx)
            x_src_1 = min(nx, nx - dx)
            y_src_0 = max(0, -dy)
            y_src_1 = min(ny, ny - dy)
            dilated[x_src_0 + dx : x_src_1 + dx, y_src_0 + dy : y_src_1 + dy] |= mask[x_src_0:x_src_1, y_src_0:y_src_1]
    return dilated


def find_connected_regions(mask, x_edges, y_edges, prof_vals, height_single_cam=2304.0):
    """
    Raggruppa i bin vetati adiacenti (8-connettività) in regioni distinte
    per stampare un riepilogo leggibile per camera e coordinate pixel.
    """
    nx, ny = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []

    for ix in range(nx):
        for iy in range(ny):
            if not mask[ix, iy] or visited[ix, iy]:
                continue

            queue = deque([(ix, iy)])
            visited[ix, iy] = True
            bins = []

            while queue:
                cx, cy = queue.popleft()
                bins.append((cx, cy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx_i, ny_i = cx + dx, cy + dy
                        if 0 <= nx_i < nx and 0 <= ny_i < ny:
                            if mask[nx_i, ny_i] and not visited[nx_i, ny_i]:
                                visited[nx_i, ny_i] = True
                                queue.append((nx_i, ny_i))

            xs = [b[0] for b in bins]
            ys = [b[1] for b in bins]
            x_min_px = float(x_edges[min(xs)])
            x_max_px = float(x_edges[max(xs) + 1])
            y_min_px = float(y_edges[min(ys)])
            y_max_px = float(y_edges[max(ys) + 1])
            y_center = 0.5 * (y_min_px + y_max_px)
            cam_id = int(y_center // height_single_cam)
            peak_adc = float(np.max([prof_vals[bx, by] for bx, by in bins]))

            regions.append({
                "id": len(regions) + 1,
                "cam": cam_id,
                "n_bins": len(bins),
                "x_min_px": x_min_px,
                "x_max_px": x_max_px,
                "y_min_px": y_min_px,
                "y_max_px": y_max_px,
                "y_min_local": y_min_px - cam_id * height_single_cam,
                "y_max_local": y_max_px - cam_id * height_single_cam,
                "peak_adc": peak_adc,
            })

    regions.sort(key=lambda r: r["peak_adc"], reverse=True)
    for idx, r in enumerate(regions, 1):
        r["id"] = idx
    return regions


def generate_topo_veto(
    input_root_file,
    source="midas",
    veto_output="gem_spark_veto.npz",
    plot_output="gem_spark_veto_check.png",
    nsigma_spark=8.0,
    ratio_threshold=None,
    abs_threshold=None,
    dilate_bins=1,
    min_occ=1,
):
    print(f"\n========================================================")
    print(f"RICERCA REGIONI DI SCARICA GEM (VETO TOPOLOGICO)")
    print(f"========================================================")
    print(f"--> Mappa di riferimento: {input_root_file}")

    with uproot.open(input_root_file) as f:
        if source == "midas":
            prof_key = "prof2d_images" if "prof2d_images" in f else "prof2d"
            occ_key = "occ2d_images" if "occ2d_images" in f else "occ2d"
        else:
            prof_key = "prof2d"
            occ_key = "occ2d"

        prof_vals, x_edges, y_edges = f[prof_key].to_numpy()
        occ_vals, _, _ = f[occ_key].to_numpy()

    active = occ_vals >= min_occ
    if not np.any(active):
        raise RuntimeError(f"Nessun bin attivo trovato in {input_root_file} con occ >= {min_occ}.")

    prof_active = prof_vals[active]
    median_val = float(np.median(prof_active))
    mad_val = float(np.median(np.abs(prof_active - median_val)))
    robust_std = 1.4826 * mad_val
    if robust_std <= 0:
        q75, q25 = np.percentile(prof_active, [75, 25])
        robust_std = float((q75 - q25) / 1.349) if q75 > q25 else float(np.std(prof_active))

    if abs_threshold is not None:
        threshold = float(abs_threshold)
        rule_str = f"soglia assoluta > {threshold:.1f} ADC/px"
    elif ratio_threshold is not None:
        threshold = float(ratio_threshold) * median_val
        rule_str = f"{ratio_threshold:.1f} x mediana ({threshold:.1f} ADC/px)"
    else:
        threshold = median_val + float(nsigma_spark) * robust_std
        rule_str = f"mediana + {nsigma_spark:.1f} * sigma_robusta ({threshold:.1f} ADC/px)"

    core_mask = active & (prof_vals > threshold)
    veto_mask = binary_dilate_2d(core_mask, radius=dilate_bins)

    n_active = int(np.sum(active))
    n_core = int(np.sum(core_mask))
    n_vetoed = int(np.sum(veto_mask))
    pct_removed = 100.0 * n_vetoed / max(n_active, 1)

    max_before = float(np.max(prof_active))
    clean_active = active & (~veto_mask)
    max_after = float(np.max(prof_vals[clean_active])) if np.any(clean_active) else 0.0
    mean_before = float(np.mean(prof_active))
    mean_after = float(np.mean(prof_vals[clean_active])) if np.any(clean_active) else 0.0

    print(f"--> Statistiche mappa bare (bin attivi={n_active}):")
    print(f"    Mediana = {median_val:.2f} ADC/px | Sigma robusta = {robust_std:.2f} ADC/px")
    print(f"    Criterio applicato: {rule_str}")
    print(f"    Bin core scariche: {n_core} | Dopo dilatazione (r={dilate_bins} bin): {n_vetoed} ({pct_removed:.2f}% area attiva)")
    print(f"    Max ADC/px:  {max_before:.1f} -> {max_after:.1f} (dopo veto)")
    print(f"    Mean ADC/px: {mean_before:.2f} -> {mean_after:.2f} (dopo veto)")

    regions = find_connected_regions(veto_mask, x_edges, y_edges, prof_vals)
    if regions:
        print(f"\n--> Individuate {len(regions)} regioni connesse di scarica GEM:")
        for r in regions:
            print(
                f"    #{r['id']:02d} | CAM {r['cam']} | X=[{r['x_min_px']:.0f}, {r['x_max_px']:.0f}] px | "
                f"Y_glob=[{r['y_min_px']:.0f}, {r['y_max_px']:.0f}] px (Y_cam=[{r['y_min_local']:.0f}, {r['y_max_local']:.0f}]) | "
                f"bins={r['n_bins']} | peak={r['peak_adc']:.1f} ADC/px"
            )
    else:
        print("\n--> Nessuna regione sopra soglia individuata con i parametri attuali.")

    # Salva maschera topologica su file .npz
    os.makedirs(os.path.dirname(os.path.abspath(veto_output)), exist_ok=True)
    np.savez_compressed(
        veto_output,
        veto_mask=veto_mask.astype(bool),
        core_mask=core_mask.astype(bool),
        x_edges=x_edges,
        y_edges=y_edges,
        threshold=np.float64(threshold),
        median_val=np.float64(median_val),
        robust_std=np.float64(robust_std),
        source=np.array(source),
        input_file=np.array(os.path.abspath(input_root_file)),
    )
    print(f"\n--> Maschera di veto topologico salvata in: {veto_output}")

    # Plot di ispezione a 3 pannelli
    prof_bare_plot = np.where(active, prof_vals, np.nan).T
    prof_clean_plot = np.where(clean_active, prof_vals, np.nan).T
    veto_overlay = np.where(veto_mask, 1.0, np.nan).T

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), dpi=150)

    # Pannello 1: Mappa bare
    m0 = axes[0].pcolormesh(x_edges, y_edges, prof_bare_plot, shading="flat", cmap="rainbow")
    cb0 = fig.colorbar(m0, ax=axes[0], pad=0.02, fraction=0.046)
    cb0.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[0].set_title(f"1. Mappa BARE\n(max={max_before:.0f}, mean={mean_before:.1f})", fontsize=11, fontweight="bold")

    # Pannello 2: Maschera di veto evidenziata
    axes[1].pcolormesh(x_edges, y_edges, prof_clean_plot, shading="flat", cmap="Greys", alpha=0.55)
    axes[1].pcolormesh(x_edges, y_edges, veto_overlay, shading="flat", cmap="autumn", vmin=0, vmax=1, alpha=0.9)
    for r in regions:
        rect = Rectangle(
            (r["x_min_px"], r["y_min_px"]),
            r["x_max_px"] - r["x_min_px"],
            r["y_max_px"] - r["y_min_px"],
            linewidth=1.4,
            edgecolor="red",
            facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].text(
            r["x_max_px"] + 40,
            0.5 * (r["y_min_px"] + r["y_max_px"]),
            f"#{r['id']} (CAM{r['cam']})",
            color="darkred",
            fontsize=8,
            fontweight="bold",
            va="center",
        )
    axes[1].set_title(
        f"2. Veto Topologico ({len(regions)} regioni)\n{n_vetoed} bin ({pct_removed:.2f}% area) | thr={threshold:.1f}",
        fontsize=11,
        fontweight="bold",
    )

    # Pannello 3: Mappa pulita post-veto
    m2 = axes[2].pcolormesh(x_edges, y_edges, prof_clean_plot, shading="flat", cmap="rainbow")
    cb2 = fig.colorbar(m2, ax=axes[2], pad=0.02, fraction=0.046)
    cb2.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[2].set_title(f"3. Mappa PULITA post-veto\n(max={max_after:.0f}, mean={mean_after:.1f})", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("X [pixel]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    axes[0].set_ylabel("Y [pixel]", fontsize=10)

    fig.suptitle(f"Controllo Veto Scariche GEM — {os.path.basename(input_root_file)}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(plot_output)), exist_ok=True)
    plt.savefig(plot_output, bbox_inches="tight", pad_inches=0.1)
    base_no_ext, _ = os.path.splitext(plot_output)
    plt.savefig(f"{base_no_ext}.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"--> Plot di controllo salvato in: {plot_output} (e {base_no_ext}.pdf)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Individua le regioni di scarica GEM da una mappa bare e crea la maschera di veto topologico.")
    parser.add_argument("--input", type=str, required=True, help="File ROOT della mappa bare del run di riferimento")
    parser.add_argument("--source", type=str, choices=["midas", "reco"], default="midas", help="Sorgente mappa: midas (default) o reco")
    parser.add_argument("--veto-output", type=str, default="gem_spark_veto.npz", help="File .npz di output con la maschera booleana")
    parser.add_argument("--plot-output", type=str, default="gem_spark_veto_check.png", help="Immagine di controllo a 3 pannelli")
    parser.add_argument("--nsigma-spark", type=float, default=8.0, help="Soglia in sigma robuste sopra la mediana (default: 8.0)")
    parser.add_argument("--ratio-threshold", type=float, default=None, help="Soglia alternativa come multiplo della mediana (es. 5.0)")
    parser.add_argument("--abs-threshold", type=float, default=None, help="Soglia assoluta alternativa in ADC/pixel")
    parser.add_argument("--dilate", type=int, default=1, help="Raggio di dilatazione in bin attorno alle scariche (default: 1)")
    parser.add_argument("--min-occ", type=int, default=1, help="Occupanza minima nel run bare (default: 1)")

    args = parser.parse_args()

    generate_topo_veto(
        input_root_file=args.input,
        source=args.source,
        veto_output=args.veto_output,
        plot_output=args.plot_output,
        nsigma_spark=args.nsigma_spark,
        ratio_threshold=args.ratio_threshold,
        abs_threshold=args.abs_threshold,
        dilate_bins=args.dilate,
        min_occ=args.min_occ,
    )