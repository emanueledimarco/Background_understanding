#!/usr/bin/env python3
"""
Orchestratore principale CYGNO.
Esegue in sequenza:
1. Creazione mappe 2D di luce (make_2dmaps_fromclusters_s3.py) per ciascun run
2. Calcolo integrale e media di luce (compute_xy_mean_from_root)
3. Generazione storico temporale e grafico trend (plot_xy_mean_vs_time.py)

Supporta il salto dei run già processati e il salvataggio degli errori in pipeline_errors.log.
"""
import sys
import os

# Aggiunge commissioning al path ed esegue il modulo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "commissioning"))
from muonmaps_lowexposure_fromclusters import run_map_processing, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestratore CYGNO: Mappe di luce, Integrali e History vs Tempo.")
    parser.add_argument("--csv", type=str, default="commissioning/runlist_stopped_muons.csv", help="Path del file CSV dei run (default: commissioning/runlist_stopped_muons.csv)")
    parser.add_argument("--run-min", type=int, default=124905, help="Run iniziale (default: 124905)")
    parser.add_argument("--run-max", type=int, default=124926, help="Run finale (default: 124926)")
    parser.add_argument("--run-description", type=str, nargs="*", default=None, help="Filtro per descrizione/i del run")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Tag S3 (default: LNGS)")
    parser.add_argument("--mode", type=str, choices=["per-run", "consecutive"], default="per-run", help="Modalità: per-run (default, 1 run per punto) o consecutive")
    parser.add_argument("--execute", action="store_true", help="Avvia l'elaborazione effettiva (default: False)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Job paralleli di elaborazione run (default: 1)")
    parser.add_argument("-w", "--workers-per-run", type=int, default=4, help="Worker di decompressione ROOT per singolo run (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Sovrascrive le mappe esistenti (default: False)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory di output (default: .)")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv", help="CSV di riepilogo con medie e tempi")
    parser.add_argument("--plot-output", type=str, default="trend_integral.png", help="Immagine del grafico temporale")
    parser.add_argument("--plot-2dmaps", action="store_true", help="Genera anche i plot 2D (PNG/PDF) per ciascun run")
    parser.add_argument("--plots-2d-dir", type=str, default="plots_2d", help="Directory di output per le mappe 2D")
    parser.add_argument("--error-log", type=str, default="pipeline_errors.log", help="File di log errori")
    parser.add_argument("--time-col", type=str, default=None, help="Colonna start-time (auto-rilevata se non specificata)")
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours", help="Unità asse temporale (default: hours)")

    args = parser.parse_args()

    run_desc = args.run_description
    if run_desc is not None and len(run_desc) == 1:
        if "," in run_desc[0]:
            run_desc = [d.strip() for d in run_desc[0].split(",")]
        else:
            run_desc = run_desc[0]

    run_map_processing(
        csv_file=args.csv,
        run_min=args.run_min,
        run_max=args.run_max,
        run_description=run_desc,
        file_s3_tag=args.file_s3_tag,
        mode=args.mode,
        execute=args.execute,
        n_jobs=args.jobs,
        workers_per_run=args.workers_per_run,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        plot_output=args.plot_output,
        plot_2dmaps=args.plot_2dmaps,
        plots_2d_dir=args.plots_2d_dir,
        error_log=args.error_log,
        time_col=args.time_col,
        time_unit=args.time_unit,
    )
