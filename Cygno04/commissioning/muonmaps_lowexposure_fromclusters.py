import os
import sys
import argparse
import subprocess
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


def get_map_filename(run_min, run_max, vgem, vdrift):
    return f"cygno04_sideA_muons_withVignetting_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"


def find_consecutive_run_ranges(df):
    """
    Raggruppa run consecutivi che hanno gli stessi valori di GEM1_V, DRIFT_V e run_description.
    """
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)

    ranges = []
    current_start = df.loc[0, "run_number"]
    current_end = df.loc[0, "run_number"]
    current_vgem = df.loc[0, "GEM1_V"]
    current_vdrift = df.loc[0, "DRIFT_V"]
    current_desc = df.loc[0, "run_description"] if "run_description" in df.columns else ""

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]
        desc = row["run_description"] if "run_description" in df.columns else ""

        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift) and (desc == current_desc):
            current_end = run_num
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
                "run_description": str(current_desc),
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift
            current_desc = desc

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
        "run_description": str(current_desc),
    })

    return ranges


def get_per_run_ranges(df):
    """
    Restituisce ogni singolo run come range autonomo [run, run].
    Consente di monitorare l'evoluzione temporale a grana fine (1 run / 10 o 20 min).
    """
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)
    ranges = []
    for _, row in df.iterrows():
        run_num = int(row["run_number"])
        vgem = int(row["GEM1_V"]) if pd.notnull(row["GEM1_V"]) else row["GEM1_V"]
        vdrift = int(row["DRIFT_V"]) if pd.notnull(row["DRIFT_V"]) else row["DRIFT_V"]
        desc = str(row["run_description"]) if "run_description" in df.columns else ""
        ranges.append({
            "run_min": run_num,
            "run_max": run_num,
            "vgem": vgem,
            "vdrift": vdrift,
            "run_description": desc,
        })
    return ranges


def log_error_entry(error_log_path, run_min, run_max, cmd, error_msg, stdout=None, stderr=None):
    """
    Scrive un log dettagliato dell'errore su file.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(error_log_path)), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(error_log_path, "a", encoding="utf-8") as f_err:
            f_err.write(f"\n{'='*70}\n")
            f_err.write(f"[{timestamp}] ERRORE SUL RUN RANGE: {run_min}-{run_max}\n")
            f_err.write(f"Comando: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
            f_err.write(f"Dettaglio: {error_msg}\n")
            if stdout:
                f_err.write(f"--- STDOUT ---\n{stdout}\n")
            if stderr:
                f_err.write(f"--- STDERR ---\n{stderr}\n")
            f_err.write(f"{'='*70}\n")
    except Exception as e:
        print(f"[CRITICAL] Impossibile scrivere sull'error log {error_log_path}: {e}")


def execute_command(task):
    """
    Worker per l'esecuzione del task su singolo core CPU.
    Cattura l'output e registra gli errori senza bloccare l'intero pool.
    """
    run_min, run_max, vgem, vdrift, cmd, error_log_path = task
    cmd_str = " ".join(cmd)
    print(f"\n[AVVIO] Range {run_min}-{run_max} (VGEM={vgem}V, VD={vdrift}V)\nComando: {cmd_str}")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            print(f"[COMPLETATO] Range {run_min}-{run_max}")
            return True, run_min, run_max, None
        else:
            err_msg = f"Processo terminato con codice di errore {proc.returncode}"
            print(f"[ERRORE] Fallito range {run_min}-{run_max}: codice {proc.returncode}")
            log_error_entry(error_log_path, run_min, run_max, cmd, err_msg, proc.stdout, proc.stderr)
            return False, run_min, run_max, err_msg
    except Exception as e:
        err_msg = str(e)
        print(f"[ECCEZIONE] Range {run_min}-{run_max}: {err_msg}")
        log_error_entry(error_log_path, run_min, run_max, cmd, err_msg)
        return False, run_min, run_max, err_msg


def run_map_processing(
    csv_file,
    run_min=None,
    run_max=None,
    run_description=None,
    file_s3_tag="LNGS",
    mode="per-run",
    execute=False,
    n_jobs=1,
    workers_per_run=4,
    overwrite=False,
    output_dir=".",
    summary_csv="xy_mean_summary.csv",
    plot_output="trend_integral.png",
    plot_2dmaps=False,
    plots_2d_dir="plots_2d",
    error_log="pipeline_errors.log",
    time_col=None,
    time_unit="hours",
):
    print(f"\n========================================================")
    print(f"CYGNO MAP & HISTORY ORCHESTRATOR")
    print(f"========================================================")
    print(f"--> File CSV di input: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File CSV non trovato: {csv_file}")

    df = pd.read_csv(csv_file)
    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()
    df["run_number"] = pd.to_numeric(df["run_number"], errors="coerce")
    df["GEM1_V"] = pd.to_numeric(df["GEM1_V"], errors="coerce")
    df["DRIFT_V"] = pd.to_numeric(df["DRIFT_V"], errors="coerce")

    # Filtro file_s3_tag
    mask = pd.Series(True, index=df.index)
    if file_s3_tag:
        target_tag = file_s3_tag.strip()
        if run_min is not None or run_max is not None:
            mask &= (df["file_s3_tag"] == target_tag) | (df["file_s3_tag"].isna()) | (df["file_s3_tag"] == "") | (df["file_s3_tag"] == "nan")
        else:
            mask &= (df["file_s3_tag"] == target_tag)

    # Filtro run_min e run_max
    if run_min is not None:
        mask &= (df["run_number"] >= run_min)
    if run_max is not None:
        mask &= (df["run_number"] <= run_max)

    # Filtro run_description
    if run_description is not None:
        if isinstance(run_description, str):
            if run_description.lower() == "all":
                pass
            elif "," in run_description:
                descs = [d.strip() for d in run_description.split(",") if d.strip()]
                mask &= df["run_description"].isin(descs)
            else:
                mask &= (df["run_description"] == run_description.strip())
        elif isinstance(run_description, (list, tuple)):
            descs = [str(d).strip() for d in run_description if str(d).strip()]
            mask &= df["run_description"].isin(descs)

    df_filtered = df[mask].copy().sort_values("run_number").reset_index(drop=True)
    print(f"--> Trovati {len(df_filtered)} run corrispondenti ai filtri:")
    if run_min or run_max:
        print(f"    Intervallo run: [{run_min} - {run_max}]")
    if run_description:
        print(f"    Descrizioni: {run_description}")

    if df_filtered.empty:
        print("[WARNING] Nessun run trovato con i criteri specificati. Uscita.")
        return

    # Determinazione dei range di esecuzione
    if mode == "consecutive":
        run_ranges = find_consecutive_run_ranges(df_filtered)
    else:
        run_ranges = get_per_run_ranges(df_filtered)

    print(f"\n========================================================")
    print(f"MODALITÀ: {mode.upper()} | {len(run_ranges)} RANGE IDENTIFICATI")
    print(f"========================================================")

    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_script = os.path.join(base_dir, "make_2dmaps_fromclusters_s3.py")
    plot_script = os.path.join(base_dir, "plot_xy_mean_vs_time.py")

    error_log_path = os.path.abspath(error_log)

    tasks_to_run = []
    skipped_count = 0

    for r in run_ranges:
        r_min = r["run_min"]
        r_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]
        desc = r.get("run_description", "")

        filename = get_map_filename(r_min, r_max, vgem, vdrift)
        output_filename = os.path.join(output_dir, filename)

        already_exists = os.path.exists(output_filename) and os.path.getsize(output_filename) > 0

        if already_exists and not overwrite:
            size_mb = os.path.getsize(output_filename) / (1024 * 1024)
            print(f"  > [SKIP - GIÀ PRESENTE] Run {r_min:6d} ({size_mb:4.1f} MB) | {desc} -> {filename}")
            skipped_count += 1
            continue

        cmd = [
            sys.executable, map_script,
            "--run-start", str(r_min),
            "--run-end", str(r_max),
            "--output", output_filename,
            "--workers", str(workers_per_run),
        ]
        tasks_to_run.append((r_min, r_max, vgem, vdrift, cmd, error_log_path))
        print(f"  > [DA GENERARE]         Run {r_min:6d} | {desc} -> {filename}")

    print(f"\nRiepilogo preparazione:")
    print(f"  - Totale run identificati: {len(run_ranges)}")
    print(f"  - Già processati (saltati): {skipped_count}")
    print(f"  - Da processare ora:       {len(tasks_to_run)}")

    if not execute:
        print(f"\n(Modalità DRY-RUN completata. Usa --execute per avviare l'elaborazione)")
        return

    failed_tasks = []
    succeeded_tasks = []

    # Step 1: Esecuzione delle mappe mancanti
    if tasks_to_run:
        max_cores = multiprocessing.cpu_count()
        actual_jobs = len(tasks_to_run) if n_jobs <= 0 else min(n_jobs, len(tasks_to_run))
        print(f"\n---> Esecuzione di {len(tasks_to_run)} task con {actual_jobs} processi paralleli...")

        if actual_jobs == 1:
            for task in tasks_to_run:
                success, r_min, r_max, err = execute_command(task)
                if success:
                    succeeded_tasks.append((r_min, r_max))
                else:
                    failed_tasks.append((r_min, r_max, err))
        else:
            with ProcessPoolExecutor(max_workers=actual_jobs) as executor:
                futures = [executor.submit(execute_command, task) for task in tasks_to_run]
                for future in as_completed(futures):
                    success, r_min, r_max, err = future.result()
                    if success:
                        succeeded_tasks.append((r_min, r_max))
                    else:
                        failed_tasks.append((r_min, r_max, err))
    else:
        print("\n---> Tutte le mappe ROOT sono già presenti.")

    # Step 2: Generazione del riepilogo e del plot temporale (History)
    print(f"\n========================================================")
    print(f"STEP 2: AGGIORNAMENTO STORICO & PLOT TEMPORALE")
    print(f"========================================================")
    plot_cmd = [
        sys.executable, plot_script,
        "--csv", csv_file,
        "--mode", mode,
        "--output-dir", output_dir,
        "--summary-csv", summary_csv,
        "--plot-output", plot_output,
        "--time-unit", time_unit,
    ]
    if file_s3_tag:
        plot_cmd.extend(["--file-s3-tag", file_s3_tag])
    if run_min is not None:
        plot_cmd.extend(["--run-min", str(run_min)])
    if run_max is not None:
        plot_cmd.extend(["--run-max", str(run_max)])
    if run_description:
        if isinstance(run_description, (list, tuple)):
            plot_cmd.extend(["--run-description"] + [str(d) for d in run_description])
        else:
            plot_cmd.extend(["--run-description", str(run_description)])
    if time_col is not None:
        plot_cmd.extend(["--time-col", time_col])

    print(f"Comando plot: {' '.join(plot_cmd)}")
    try:
        subprocess.run(plot_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] Generazione plot fallita: {e}")
        log_error_entry(error_log_path, run_min or 0, run_max or 0, plot_cmd, str(e))

    # Step 3: Generazione opzionale mappe 2D prof2d (PNG/PDF)
    if plot_2dmaps:
        print(f"\n========================================================")
        print(f"STEP 3: GENERAZIONE MAPPE 2D (PNG/PDF)")
        print(f"========================================================")
        os.makedirs(plots_2d_dir, exist_ok=True)
        plot2d_script = os.path.join(base_dir, "plot_2dmaps.py")
        for r in run_ranges:
            r_min = r["run_min"]
            r_max = r["run_max"]
            vgem = r["vgem"]
            vdrift = r["vdrift"]
            desc = r.get("run_description", "")
            root_fname = get_map_filename(r_min, r_max, vgem, vdrift)
            root_fpath = os.path.join(output_dir, root_fname)
            if not (os.path.exists(root_fpath) and os.path.getsize(root_fpath) > 0):
                continue
            title = f"Run {r_min} - {desc} (VGEM={vgem}V, VD={vdrift}V)"
            out_name = f"run_{r_min}" if r_min == r_max else f"run_{r_min}_{r_max}"
            p2d_cmd = [
                sys.executable, plot2d_script,
                "--from-reco",
                "--input", root_fpath,
                "--output-dir", plots_2d_dir,
                "--output-name", out_name,
                "--zmin-prof", "0",
                "--zmax-prof", "8",
                "--title", title
            ]
            try:
                subprocess.run(p2d_cmd, check=True, stdout=subprocess.DEVNULL)
                print(f"  > Mappa 2D generata: {plots_2d_dir}/prof2d_{out_name}.png")
            except Exception as e:
                print(f"  [ERRORE] Mappa 2D fallita per {out_name}: {e}")

    # Stampa report finale
    print(f"\n========================================================")
    print(f"REPORT FINALE PIPELINE")
    print(f"========================================================")
    print(f"  - Run totali considerati: {len(run_ranges)}")
    print(f"  - Run già esistenti:      {skipped_count}")
    print(f"  - Run generati con successo: {len(succeeded_tasks)}")
    print(f"  - Run falliti:            {len(failed_tasks)}")
    if failed_tasks:
        print(f"\n  [ATTENZIONE] I seguenti run hanno generato errori:")
        for r_min, r_max, err in failed_tasks:
            print(f"    * Run {r_min}-{r_max}: {err}")
        print(f"  --> Log completo degli errori salvato in: {error_log_path}")
    print(f"  --> File summary CSV: {os.path.abspath(summary_csv)}")
    print(f"  --> Grafico history:  {os.path.abspath(plot_output)}")
    print(f"========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestratore CYGNO: mappe di luce medie, integrale e history vs tempo.")
    parser.add_argument("--csv", type=str, default="commissioning/runlist_stopped_muons.csv", help="Path del file CSV dei run (default: commissioning/runlist_stopped_muons.csv)")
    parser.add_argument("--run-min", type=int, default=124905, help="Run minimo da processare (default: 124905)")
    parser.add_argument("--run-max", type=int, default=124926, help="Run massimo da processare (default: 124926)")
    parser.add_argument("--run-description", type=str, nargs="*", default=None, help="Descrizione/i dei run da filtrare (default: tutte)")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Tag S3 da selezionare (default: LNGS)")
    parser.add_argument("--mode", type=str, choices=["per-run", "consecutive"], default="per-run", help="Modalità: 'per-run' per evoluzione temporale (default) o 'consecutive' per fondere run adiacenti")
    parser.add_argument("--execute", action="store_true", help="Esegue effettivamente l'elaborazione (se omesso: dry-run)")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Numero di job paralleli (default: 1)")
    parser.add_argument("-w", "--workers-per-run", type=int, default=4, help="Numero di worker decompressione ROOT per singolo run (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Rigenera i file ROOT anche se esistono già")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory di output per i file ROOT delle mappe (default: .)")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv", help="CSV di riepilogo con medie e tempi")
    parser.add_argument("--plot-output", type=str, default="trend_integral.png", help="Nome file immagine del grafico")
    parser.add_argument("--plot-2dmaps", action="store_true", help="Genera anche i plot 2D (PNG/PDF) di prof2d per ciascun run")
    parser.add_argument("--plots-2d-dir", type=str, default="plots_2d", help="Directory per i plot 2D (default: plots_2d)")
    parser.add_argument("--error-log", type=str, default="pipeline_errors.log", help="Path del file di log per gli errori (default: pipeline_errors.log)")
    parser.add_argument("--time-col", type=str, default=None, help="Nome colonna timestamp (opzionale)")
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours", help="Unità asse X (default: hours)")

    args = parser.parse_args()

    # Gestione run_description (stringa singola, virgole o lista)
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
