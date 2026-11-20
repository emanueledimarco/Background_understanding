import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_2d_maps(
    input_root_file,
    output_dir="./plots",
    output_name=None,
    use_log=False,
    zmin_prof=None, zmax_prof=None,
    origin='images',
    title=None
):
    with uproot.open(input_root_file) as f:
        if origin=='images':
            occ2d_hist = f["occ2d_images"]
            prof2d_hist = f["prof2d_images"]
        else:
            occ2d_hist = f["occ2d"]
            prof2d_hist = f["prof2d"]            

        occ_vals, xedges, yedges = occ2d_hist.to_numpy()
        prof_vals, _, _ = prof2d_hist.to_numpy()

    # Trasposta per allineare (nx, ny) di Uproot a (ny, nx) di Matplotlib
    occ_vals = occ_vals.T
    prof_vals = prof_vals.T

    # Masking dei bin vuoti (occupancy = 0) -> diventano trasparenti/bianchi
    zero_mask = (occ_vals == 0)
    prof_masked = np.where(zero_mask, np.nan, prof_vals)

    COLORMAP = "rainbow"

    os.makedirs(output_dir, exist_ok=True)

    # Definizione del nome base del file di output
    base_filename = f"prof2d_{output_name}" if output_name else "prof2d"
    output_path = os.path.join(output_dir, base_filename)

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(7, 10))

    norm = None
    if use_log:
        vmin = zmin_prof if (zmin_prof is not None and zmin_prof > 0) else np.nanmin(prof_masked)
        vmax = zmax_prof if zmax_prof is not None else np.nanmax(prof_masked)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = zmin_prof
        vmax = zmax_prof

    mesh = ax.pcolormesh(
        xedges, yedges, prof_masked,
        shading='flat',
        cmap=COLORMAP,
        vmin=vmin if not use_log else None,
        vmax=vmax if not use_log else None,
        norm=norm
    )

    cbar = fig.colorbar(mesh, ax=ax, pad=0.03, fraction=0.046)
    cbar.set_label("Mean Charge / Hit [ADC]", fontsize=12)

    plot_title = title if title else "Mean Amplitude Profile 2D"
    ax.set_title(plot_title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("X [pixel]", fontsize=12)
    ax.set_ylabel("Y [pixel]", fontsize=12)

    # Mantiene le proporzioni geometriche reali dei pixel
    ax.set_aspect('equal')
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    # Salva in formato PDF e PNG rimuovendo lo spazio bianco superfluo attorno
    plt.savefig(f"{output_path}.pdf", format="pdf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{output_path}.png", format="png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Grafico salvato in: {output_path}.pdf e {output_path}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 2D del solo prof2d per le mappe CYGNO.")
    parser.add_argument("--input", type=str, required=True, help="Path del file ROOT di input")
    parser.add_argument("--output-dir", type=str, default="./plots", help="Directory di output per le immagini")
    parser.add_argument("--output-name", type=str, default=None, help="Nome identificativo per l'output (es. 'run20000' genera 'prof2d_run20000.pdf/png')")
    parser.add_argument("--log", action="store_true", help="Usa scala logaritmica per l'asse Z")
    parser.add_argument("--from-reco", action="store_true", help="Usa l'output delle mappe fatte dai redpix dei cluster RECO")

    # Range asse Z per il Profilo 2D
    parser.add_argument("--zmin-prof", type=float, default=None, help="Z min per Profilo 2D")
    parser.add_argument("--zmax-prof", type=float, default=None, help="Z max per Profilo 2D")
    parser.add_argument("--title", type=str, default=None, help="Titolo del grafico (opzionale)")

    args = parser.parse_args()

    origin = 'reco' if args.from_reco else 'images'
    
    plot_2d_maps(
        input_root_file=args.input,
        output_dir=args.output_dir,
        output_name=args.output_name,
        use_log=args.log,
        zmin_prof=args.zmin_prof,
        zmax_prof=args.zmax_prof,
        origin=origin,
        title=args.title
    )
