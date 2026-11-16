import os
import sys
import argparse
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


def get_map_filename(run_min, run_max, vgem, vdrift):
    return f"cygno04_sideA_muons_withVignetting_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"


def find_consecutive_run_ranges(df):
    """
    Raggruppa run consecutivi che hanno gli stessi valori di GEM1_V e DRIFT_V.
    """
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)

    ranges = []
    current_start = df.loc[0, "run_number"]
    current_end = df.loc[0, "run_number"]
    current_vgem = df.loc[0, "GEM1_V"]
    current_vdrift = df.loc[0, "DRIFT_V"]

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]

        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift):
            current_end = run_num
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
    })

    return ranges


def execute_command(task):
    """
    Funzione worker eseguita in parallelo da ciascun processo CPU.
    """
    run_min, run_max, vgem, vdrift, cmd = task
    cmd_str = " ".join(cmd)
    print(f"\n[AVVIO CORE] Range {run_min}-{run_max} (VGEM={vgem}V, VD={vdrift}V)\nComando: {cmd_str}")

    try:
        subprocess.run(cmd, check=True)
        print(f"[COMPLETATO] Range {run_min}-{run_max}")
        return True, run_min, run_max
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] Fallito range {run_min}-{run_max}: {e}")
        return False, run_min, run_max


def run_map_processing(
    csv_file,
    run_description="MuonMap",
    file_s3_tag="LNGS",
    execute=False,
    n_jobs=1,
    overwrite=False,
    output_dir=".",
    summary_csv="xy_mean_summary.csv",
    plot_output="xy_mean_vs_time.png",
    time_col=None,
    time_unit="hours",
):
    print(f"--> Lettura file CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()

    target_desc = run_description.strip()
    target_tag = file_s3_tag.strip()

    mask = (df["run_description"] == target_desc) & (df["file_s3_tag"] == target_tag)
    df_filtered = df[mask].copy()

    print(f"--> Trovati {len(df_filtered)} run con run_description='{target_desc}' e file_s3_tag='{target_tag}'.")

    if df_filtered.empty:
        print("Nessun run trovato con i criteri specificati. Uscita.")
        return

    df_filtered["run_number"] = pd.to_numeric(df_filtered["run_number"], errors="coerce")
    df_filtered["GEM1_V"] = pd.to_numeric(df_filtered["GEM1_V"], errors="coerce")
    df_filtered["DRIFT_V"] = pd.to_numeric(df_filtered["DRIFT_V"], errors="coerce")

    run_ranges = find_consecutive_run_ranges(df_filtered)

    print(f"\n========================================================")
    print(f"IDENTIFICATI {len(run_ranges)} RUN RANGES CONSECUTIVI")
    print(f"========================================================")

    os.makedirs(output_dir, exist_ok=True)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_script = os.path.join(base_dir, "make_2dmaps_fromclusters_s3.py")
    plot_script = os.path.join(base_dir, "plot_xy_mean_vs_time.py")

    tasks_to_run = []
    for r in run_ranges:
        run_min = r["run_min"]
        run_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]

        filename = get_map_filename(run_min, run_max, vgem, vdrift)
        output_filename = os.path.join(output_dir, filename)

        already_exists = os.path.exists(output_filename) and os.path.getsize(output_filename) > 0

        if already_exists and not overwrite:
            print(f"  > [SKIP - ESISTE GIÀ] Range {run_min}-{run_max} | VGEM={vgem}V, VD={vdrift}V -> {output_filename}")
            continue

        cmd = [
            sys.executable, map_script,
            "--run-start", str(run_min),
            "--run-end", str(run_max),
            "--output", output_filename,
            "--workers", "4",
        ]
        tasks_to_run.append((run_min, run_max, vgem, vdrift, cmd))
        print(f"  > [DA GENERARE]       Range {run_min}-{run_max} | VGEM={vgem}V, VD={vdrift}V -> {output_filename}")

    if not execute:
        print(f"\n(Modalità DRY-RUN: {len(tasks_to_run)} mappe da generare, {len(run_ranges) - len(tasks_to_run)} già presenti. Usa --execute per avviare)")
        return

    # Step 1: Genera in parallelo solo le mappe mancanti
    if tasks_to_run:
        max_cores = multiprocessing.cpu_count()
        if n_jobs <= 0 or n_jobs > max_cores:
            n_jobs = max_cores

        actual_workers = min(n_jobs, len(tasks_to_run))
        print(f"\n---> ESECUZIONE PARALLELA DI {len(tasks_to_run)} TASK SU {actual_workers} CORE CPU (disponibili: {max_cores})...\n")

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = [executor.submit(execute_command, task) for task in tasks_to_run]
            for future in as_completed(futures):
                future.result()
    else:
        print("\n---> Tutti i file ROOT esistono già: salto make_2dmaps_fromclusters_s3.py.")

    # Step 2: Lancia lo script per la media in (x, y) e il plot vs tempo
    plot_cmd = [
        sys.executable, plot_script,
        "--csv", csv_file,
        "--run-description", target_desc,
        "--file-s3-tag", target_tag,
        "--output-dir", output_dir,
        "--summary-csv", summary_csv,
        "--plot-output", plot_output,
        "--time-unit", time_unit,
    ]
    if time_col is not None:
        plot_cmd.extend(["--time-col", time_col])

    print(f"\n---> Avvio analisi e plot: {' '.join(plot_cmd)}")
    subprocess.run(plot_cmd, check=True)

# history: python commissioning/muonmaps_lowexposure_fromclusters.py --csv commissioning/runlist_stopped_muons_test2runs.csv --time-unit hours --plot-output trend_integral.png --run-description "gas stopped - Fe - pos 5" --execute
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa file CSV, crea le mappe 2D mancanti e lancia il plot vs tempo.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV di input")
    parser.add_argument("--run-description", type=str, default="gas stopped - Fe - pos 5", help="Valore di run_description da selezionare (es. 'gas stopped - Fe - pos 5')")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Valore di file_s3_tag da selezionare (default: LNGS)")
    parser.add_argument("--execute", action="store_true", help="Esegue effettivamente i comandi e il plot")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Numero di job/processi paralleli (-1 per tutti i core)")
    parser.add_argument("--overwrite", action="store_true", help="Rigenera i file ROOT anche se esistono già")
    parser.add_argument("--output-dir", type=str, default=".", help="Cartella in cui salvare/cercare i file ROOT (default: .)")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv", help="CSV di output con medie e tempi")
    parser.add_argument("--plot-output", type=str, default="xy_mean_vs_time.png", help="File immagine di output per il plot")
    parser.add_argument("--time-col", type=str, default=None, help="Nome della colonna start-time nel CSV (opzionale)")
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours", help="Unità asse X (default: hours)")

    args = parser.parse_args()

    run_map_processing(
        csv_file=args.csv,
        run_description=args.run_description,
        file_s3_tag=args.file_s3_tag,
        execute=args.execute,
        n_jobs=args.jobs,
        overwrite=args.overwrite,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        plot_output=args.plot_output,
        time_col=args.time_col,
        time_unit=args.time_unit,
    )
