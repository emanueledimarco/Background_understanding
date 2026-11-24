#!/usr/bin/env python3
"""
Orchestratore principale CYGNO.
Supporta:
- Step 1 (--make-topo-veto): mappa bare sul primo run + individuazione scariche GEM
- Step 2 (--topo-veto-file): generazione mappe 2D pulite su tutti i run (con pulizia spike per singolo evento + su prof2d + taglio bordi)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muonmaps_lowexposure_fromclusters import run_map_processing, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestratore CYGNO: Mappe di luce, Veto Topologico e History vs Tempo.")
    parser.add_argument("--csv", type=str, default="commissioning/runlist_stopped_muons.csv", help="Path del file CSV dei run")
    parser.add_argument("--source", type=str, choices=["midas", "reco"], default="midas", help="Sorgente dati: midas (immagini raw, default) o reco (clusters)")
    parser.add_argument("--pedrun", type=int, default=None, help="Numero del run di piedistallo per modalità midas")
    parser.add_argument("--nsigma", type=float, default=3.0, help="Soglia in nsigma sul piedistallo per modalità midas (default: 3.0)")
    parser.add_argument("--tmpdir", type=str, default="/tmp/", help="Directory temporanea per download MIDAS (default: /tmp/)")
    parser.add_argument("--ped-dir", type=str, default="./pedestals", help="Directory cache piedistalli .npz (default: ./pedestals)")
    parser.add_argument("--veto-events-dir", type=str, default="./veto_events", help="Directory file veto spikes (default: ./veto_events)")
    parser.add_argument("--cleanup-midas", action="store_true", help="Rimuove immediatamente i file .mid.gz da tmpdir dopo l'uso")
    parser.add_argument("--max-cache-gb", type=float, default=30.0, help="Limite massimo cache .mid.gz in tmpdir in GB (default: 30.0)")
    parser.add_argument("--max-downloads", type=int, default=2, help="Download S3 simultanei massimi (default: 2)")
    parser.add_argument("--stall-timeout", type=int, default=90, help="Timeout stallo download in secondi (default: 90)")
    parser.add_argument("--device", type=str, choices=["auto", "mps", "cuda", "cpu"], default="auto", help="Device PyTorch (default: auto)")

    parser.add_argument("--make-topo-veto", action="store_true", help="Step 1: crea mappa bare dal 1° run e individua le scariche GEM")
    parser.add_argument("--veto-ref-run", type=int, default=None, help="Run specifico su cui costruire il veto topologico (default: primo run filtrato)")
    parser.add_argument("--topo-veto-file", type=str, default=None, help="File .npz della maschera di veto topologico da applicare")
    parser.add_argument("--veto-plot-output", type=str, default="gem_spark_veto_check.png", help="Plot diagnostico del veto topologico")
    parser.add_argument("--topo-spark-nsigma", type=float, default=8.0, help="Soglia in sigma robuste per il veto topologico statico (default: 8.0)")
    parser.add_argument("--edge-margin", type=float, default=100.0, help="Margine esterno in pixel da escludere su tutti e 4 i lati (default: 100.0, 0 per disabilitare)")

    parser.add_argument("--no-event-spark-veto", action="store_true", help="Disabilita la pulizia dinamica delle scariche sulle singole foto/eventi")
    parser.add_argument("--event-spark-nsigma", type=float, default=25.0, help="Soglia core in sigma robuste sul singolo evento (default: 25.0)")
    parser.add_argument("--event-spark-skirt-nsigma", type=float, default=15.0, help="Soglia minima di coda sul singolo evento (default: 15.0)")
    parser.add_argument("--event-spark-max-radius", type=int, default=5, help="Raggio massimo di crescita in bin sul singolo evento (default: 5)")
    parser.add_argument("--event-spark-hard-cap", type=float, default=400.0, help="Soglia fissa di sicurezza in ADC/pixel sul singolo evento (default: 400.0)")
    parser.add_argument("--event-spark-dilate", type=int, default=1, help="Dilatazione in bin attorno alla scarica del singolo evento (default: 1)")

    parser.add_argument("--no-dynamic-spark-veto", action="store_true", help="Disabilita la rimozione dinamica delle spike su prof2d nella history")
    parser.add_argument("--spark-nsigma", type=float, default=20.0, help="Soglia core in sigma robuste per spike dinamiche su prof2d (default: 20.0)")
    parser.add_argument("--spark-skirt-nsigma", type=float, default=12.0, help="Soglia minima di coda in sigma robuste attorno al core (default: 12.0)")
    parser.add_argument("--spark-max-radius", type=int, default=4, help="Raggio massimo di crescita in bin dal nucleo della spike su prof2d (default: 4)")
    parser.add_argument("--spark-hard-cap", type=float, default=150.0, help="Soglia alta fissa di sicurezza in ADC/pixel su prof2d (default: 150.0, <=0 per disabilitare)")
    parser.add_argument("--spark-ratio", type=float, default=None, help="Soglia alternativa come multiplo della mediana")
    parser.add_argument("--spark-abs", type=float, default=None, help="Soglia assoluta alternativa in ADC/pixel")
    parser.add_argument("--spark-dilate", type=int, default=1, help="Dilatazione in bin attorno alle scariche (default: 1)")
    parser.add_argument("--spark-check-dir", type=str, default=None, help="Directory per i plot di controllo delle spike rimosse dinamicamente")

    parser.add_argument("--run-min", type=int, default=124905, help="Run iniziale (default: 124905)")
    parser.add_argument("--run-max", type=int, default=124926, help="Run finale (default: 124926)")
    parser.add_argument("--run-description", type=str, nargs="*", default=None, help="Filtro opzionale per descrizione/i del run (default: nessun filtro)")
    parser.add_argument("--include-pedestals", action="store_true", help="Include anche i run di piedistallo se --run-description non è specificato")
    parser.add_argument("--group-by-desc", action="store_true", help="Separa le curve del grafico per run_description")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Tag S3 (default: LNGS)")
    parser.add_argument("--mode", type=str, choices=["per-run", "consecutive"], default="per-run", help="Modalità: per-run (default, 1 run per punto) o consecutive")
    parser.add_argument("--execute", action="store_true", help="Avvia l'elaborazione effettiva (default: False)")
    parser.add_argument("-j", "--jobs", type=int, default=2, help="Job paralleli di elaborazione run (default: 2)")
    parser.add_argument("-w", "--workers-per-run", type=int, default=4, help="Worker di decompressione ROOT per singolo run in modalità reco (default: 4)")
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
        source=args.source,
        pedrun=args.pedrun,
        nsigma=args.nsigma,
        tmpdir=args.tmpdir,
        ped_dir=args.ped_dir,
        veto_events_dir=args.veto_events_dir,
        cleanup_midas=args.cleanup_midas,
        max_cache_gb=args.max_cache_gb,
        max_downloads=args.max_downloads,
        stall_timeout=args.stall_timeout,
        device=args.device,
        make_topo_veto=args.make_topo_veto,
        veto_ref_run=args.veto_ref_run,
        topo_veto_file=args.topo_veto_file,
        veto_plot_output=args.veto_plot_output,
        topo_spark_nsigma=args.topo_spark_nsigma,
        edge_margin=args.edge_margin,
        event_spark_veto=not args.no_event_spark_veto,
        event_spark_nsigma=args.event_spark_nsigma,
        event_spark_skirt_nsigma=args.event_spark_skirt_nsigma,
        event_spark_max_radius=args.event_spark_max_radius,
        event_spark_hard_cap=args.event_spark_hard_cap,
        event_spark_dilate=args.event_spark_dilate,
        dynamic_spark_veto=not args.no_dynamic_spark_veto,
        spark_nsigma=args.spark_nsigma,
        spark_skirt_nsigma=args.spark_skirt_nsigma,
        spark_max_radius=args.spark_max_radius,
        spark_hard_cap=args.spark_hard_cap,
        spark_ratio=args.spark_ratio,
        spark_abs=args.spark_abs,
        spark_dilate=args.spark_dilate,
        spark_check_dir=args.spark_check_dir,
        run_min=args.run_min,
        run_max=args.run_max,
        run_description=run_desc,
        exclude_pedestals=not args.include_pedestals,
        group_by_desc=args.group_by_desc,
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