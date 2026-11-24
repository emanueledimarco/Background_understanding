import os
import sys
import time
import argparse
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


PEDESTAL_REGEX = r"(?:^|[\s_\-])ped(?:estal)?(?:$|[\s_\-])"


def get_map_filename(run_min, run_max, vgem, vdrift, source="midas", use_topoveto=False, is_bare=False):
    if is_bare:
        tag = "_bare"
    elif use_topoveto:
        tag = "_topoveto"
    else:
        tag = ""

    if source == "midas":
        return f"cygno04_sideA_midas_raw{tag}_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"
    return f"cygno04_sideA_muons_withVignetting{tag}_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"


def resolve_pedruns(df_filtered, df_full, explicit_pedrun=None):
    if explicit_pedrun is not None:
        return pd.Series(int(explicit_pedrun), index=df_filtered.index)

    ped_cols = [c for c in df_filtered.columns if c.lower() in ("pedrun", "ped_run", "pedestal_run", "pedestal")]
    for col in ped_cols:
        vals = pd.to_numeric(df_filtered[col], errors="coerce")
        if vals.notna().all():
            return vals.astype(int)

    ped_mask = df_full["run_description"].astype(str).str.contains(PEDESTAL_REGEX, case=False, regex=True, na=False)
    ped_runs = pd.to_numeric(df_full.loc[ped_mask, "run_number"], errors="coerce").dropna().astype(int).sort_values().to_numpy()

    if len(ped_runs) == 0:
        raise ValueError(
            "Impossibile determinare il pedrun automaticamente dal CSV (nessuna colonna pedestal "
            "e nessun run di piedistallo trovato). Specificalo con --pedrun <NUMERO_RUN>."
        )

    resolved = []
    for run_num in df_filtered["run_number"].astype(int):
        preceding = ped_runs[ped_runs <= run_num]
        if len(preceding) > 0:
            resolved.append(int(preceding[-1]))
        else:
            nearest_idx = int(abs(ped_runs - run_num).argmin())
            resolved.append(int(ped_runs[nearest_idx]))

    return pd.Series(resolved, index=df_filtered.index)


def find_consecutive_run_ranges(df, split_by_desc=False):
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)

    ranges = []
    current_start = df.loc[0, "run_number"]
    current_end = df.loc[0, "run_number"]
    current_vgem = df.loc[0, "GEM1_V"]
    current_vdrift = df.loc[0, "DRIFT_V"]
    first_desc = str(df.loc[0, "run_description"])
    current_descs = [first_desc] if first_desc else []
    current_ped = df.loc[0, "_pedrun"] if "_pedrun" in df.columns else None

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]
        desc = str(row["run_description"])
        ped = row["_pedrun"] if "_pedrun" in df.columns else None

        same_desc = (desc == current_descs[-1]) if (split_by_desc and current_descs) else True

        if (
            (run_num == current_end + 1)
            and (vgem == current_vgem)
            and (vdrift == current_vdrift)
            and same_desc
            and (ped == current_ped)
        ):
            current_end = run_num
            if desc and desc not in current_descs:
                current_descs.append(desc)
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
                "run_description": " | ".join(current_descs),
                "pedrun": int(current_ped) if pd.notnull(current_ped) else None,
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift
            current_descs = [desc] if desc else []
            current_ped = ped

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
        "run_description": " | ".join(current_descs),
        "pedrun": int(current_ped) if pd.notnull(current_ped) else None,
    })

    return ranges


def get_per_run_ranges(df):
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)
    ranges = []
    for _, row in df.iterrows():
        run_num = int(row["run_number"])
        vgem = int(row["GEM1_V"]) if pd.notnull(row["GEM1_V"]) else row["GEM1_V"]
        vdrift = int(row["DRIFT_V"]) if pd.notnull(row["DRIFT_V"]) else row["DRIFT_V"]
        ped = int(row["_pedrun"]) if ("_pedrun" in df.columns and pd.notnull(row["_pedrun"])) else None
        ranges.append({
            "run_min": run_num,
            "run_max": run_num,
            "vgem": vgem,
            "vdrift": vdrift,
            "run_description": str(row["run_description"]),
            "pedrun": ped,
        })
    return ranges


def execute_command(task):
    run_min, run_max, vgem, vdrift, cmd = task
    cmd_str = " ".join(cmd)

    for attempt in (1, 2):
        tag_attempt = "" if attempt == 1 else " [RETRY 2/2]"
        print(f"\n[AVVIO CORE]{tag_attempt} Range {run_min}-{run_max} (VGEM={vgem}V, VD={vdrift}V)\nComando: {cmd_str}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            print(f"[COMPLETATO] Range {run_min}-{run_max}", flush=True)
            return True, run_min, run_max, ""
        except subprocess.CalledProcessError as e:
            if attempt == 1:
                print(f"[WARNING] Range {run_min}-{run_max} interrotto ({e}), riprovo tra 3s...", flush=True)
                time.sleep(3.0)
            else:
                err_msg = f"Fallito range {run_min}-{run_max}: {e}"
                print(f"[ERRORE] {err_msg}", flush=True)
                return False, run_min, run_max, err_msg


def run_map_processing(
    csv_file,
    source="midas",
    pedrun=None,
    nsigma=3.0,
    tmpdir="/tmp/",
    ped_dir="./pedestals",
    veto_events_dir="./veto_events",
    cleanup_midas=False,
    max_cache_gb=30.0,
    max_downloads=2,
    stall_timeout=90,
    device="auto",
    make_topo_veto=False,
    veto_ref_run=None,
    topo_veto_file=None,
    veto_plot_output="gem_spark_veto_check.png",
    topo_spark_nsigma=8.0,
    edge_margin=100.0,
    event_spark_veto=True,
    event_spark_nsigma=25.0,
    event_spark_skirt_nsigma=15.0,
    event_spark_max_radius=5,
    event_spark_hard_cap=400.0,
    event_spark_dilate=1,
    dynamic_spark_veto=True,
    spark_nsigma=20.0,
    spark_skirt_nsigma=12.0,
    spark_max_radius=4,
    spark_hard_cap=150.0,
    spark_ratio=None,
    spark_abs=None,
    spark_dilate=1,
    spark_check_dir=None,
    run_min=None,
    run_max=None,
    run_description=None,
    exclude_pedestals=True,
    group_by_desc=False,
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
    print(f"--> Lettura file CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()
    df["run_number"] = pd.to_numeric(df["run_number"], errors="coerce")
    df["GEM1_V"] = pd.to_numeric(df["GEM1_V"], errors="coerce")
    df["DRIFT_V"] = pd.to_numeric(df["DRIFT_V"], errors="coerce")

    mask = pd.Series(True, index=df.index)
    if file_s3_tag:
        mask &= (df["file_s3_tag"] == file_s3_tag.strip())
    if run_min is not None:
        mask &= (df["run_number"] >= run_min)
    if run_max is not None:
        mask &= (df["run_number"] <= run_max)

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
    elif exclude_pedestals:
        is_ped = df["run_description"].str.contains(PEDESTAL_REGEX, case=False, regex=True, na=False)
        n_skipped_ped = int((mask & is_ped).sum())
        if n_skipped_ped > 0:
            print(f"--> Esclusi automaticamente {n_skipped_ped} run di piedistallo nel range (usa --include-pedestals per includerli).")
        mask &= (~is_ped)

    df_filtered = df[mask].copy().sort_values("run_number").reset_index(drop=True)
    unique_descs = df_filtered["run_description"].unique().tolist()
    print(f"--> Trovati {len(df_filtered)} run corrispondenti ai filtri (source={source}, descrizioni={unique_descs}).")

    if df_filtered.empty:
        print("Nessun run trovato con i criteri specificati. Uscita.")
        return

    if source == "midas":
        df_filtered["_pedrun"] = resolve_pedruns(df_filtered, df, explicit_pedrun=pedrun)

    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    map_script = os.path.join(base_dir, "make_2dmaps_frommidas.py" if source == "midas" else "make_2dmaps_fromclusters_s3.py")
    veto_script = os.path.join(base_dir, "find_gem_spark_veto.py")
    plot_script = os.path.join(base_dir, "plot_xy_mean_vs_time.py")
    plot_2d_script = os.path.join(base_dir, "plot_2dmaps.py")

    def build_midas_event_veto_flags():
        if not event_spark_veto:
            return ["--no-event-spark-veto"]
        return [
            "--event-spark-nsigma", str(event_spark_nsigma),
            "--event-spark-skirt-nsigma", str(event_spark_skirt_nsigma),
            "--event-spark-max-radius", str(event_spark_max_radius),
            "--event-spark-hard-cap", str(event_spark_hard_cap),
            "--event-spark-dilate", str(event_spark_dilate),
            "--edge-margin", str(edge_margin),
        ]

    if make_topo_veto:
        if veto_ref_run is not None:
            ref_rows = df_filtered[df_filtered["run_number"] == veto_ref_run]
            if ref_rows.empty:
                ref_rows = df[df["run_number"] == veto_ref_run]
            if ref_rows.empty:
                raise ValueError(f"Run di riferimento --veto-ref-run {veto_ref_run} non trovato nel CSV.")
            ref_row = ref_rows.iloc[0]
        else:
            ref_row = df_filtered.iloc[0]

        ref_run = int(ref_row["run_number"])
        vgem = int(ref_row["GEM1_V"]) if pd.notnull(ref_row["GEM1_V"]) else ref_row["GEM1_V"]
        vdrift = int(ref_row["DRIFT_V"]) if pd.notnull(ref_row["DRIFT_V"]) else ref_row["DRIFT_V"]
        r_ped = int(ref_row["_pedrun"]) if ("_pedrun" in ref_row and pd.notnull(ref_row["_pedrun"])) else pedrun

        bare_fname = get_map_filename(ref_run, ref_run, vgem, vdrift, source=source, is_bare=True)
        bare_path = os.path.join(output_dir, bare_fname)
        veto_out = topo_veto_file if topo_veto_file is not None else "gem_spark_veto.npz"

        print(f"\n========================================================")
        print(f"MODALITÀ --make-topo-veto (RUN DI RIFERIMENTO: {ref_run})")
        print(f"========================================================")
        print(f"  > Mappa BARE:      {bare_path}")
        print(f"  > Maschera VETO:   {veto_out}")
        print(f"  > Plot controllo:  {veto_plot_output}")

        if not execute:
            print("\n(Modalità DRY-RUN: usa --execute per generare la mappa bare e la maschera di veto)")
            return

        if not (os.path.exists(bare_path) and os.path.getsize(bare_path) > 0) or overwrite:
            if source == "midas":
                cmd = [
                    sys.executable, map_script,
                    "--run-start", str(ref_run),
                    "--run-end", str(ref_run),
                    "--pedrun", str(r_ped),
                    "--output", bare_path,
                    "--nsigma", str(nsigma),
                    "--tag", str(file_s3_tag),
                    "--tmpdir", tmpdir,
                    "--ped-dir", ped_dir,
                    "--veto-events-dir", veto_events_dir,
                    "--max-cache-gb", str(max_cache_gb),
                    "--max-downloads", str(max_downloads),
                    "--stall-timeout", str(stall_timeout),
                    "--device", device,
                    "--no-event-spark-veto",
                ]
                if cleanup_midas:
                    cmd.append("--cleanup-midas")
            else:
                cmd = [
                    sys.executable, map_script,
                    "--run-start", str(ref_run),
                    "--run-end", str(ref_run),
                    "--output", bare_path,
                    "--workers", str(workers_per_run),
                ]
            print(f"\n---> Generazione mappa BARE: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            print(f"\n---> Mappa BARE già esistente: {bare_path}")

        veto_cmd = [
            sys.executable, veto_script,
            "--input", bare_path,
            "--source", source,
            "--veto-output", veto_out,
            "--plot-output", veto_plot_output,
            "--nsigma-spark", str(topo_spark_nsigma),
            "--dilate", str(spark_dilate),
        ]
        if spark_ratio is not None:
            veto_cmd.extend(["--ratio-threshold", str(spark_ratio)])
        if spark_abs is not None:
            veto_cmd.extend(["--abs-threshold", str(spark_abs)])

        print(f"\n---> Ricerca regioni di scarica GEM: {' '.join(veto_cmd)}")
        subprocess.run(veto_cmd, check=True)

        print("\n========================================================")
        print("STEP 1 COMPLETATO! Controlla il grafico:")
        print(f"  {veto_plot_output}")
        print("\nSe le regioni di scarica individuate sono OK, lancia lo STEP 2 su tutti i run con:")
        print(f"  python commissioning/run_orchestrator.py --csv {csv_file} --topo-veto-file {veto_out} --execute -j 2")
        print("========================================================")
        return

    use_topoveto = topo_veto_file is not None
    if use_topoveto and not os.path.exists(topo_veto_file):
        raise FileNotFoundError(f"File --topo-veto-file non trovato: {topo_veto_file}")

    if mode == "consecutive":
        run_ranges = find_consecutive_run_ranges(df_filtered, split_by_desc=group_by_desc)
    else:
        run_ranges = get_per_run_ranges(df_filtered)

    print(f"\n========================================================")
    print(f"IDENTIFICATI {len(run_ranges)} TASK ({mode}, source={source}, topoveto={use_topoveto}, evspark={event_spark_veto}, dynspark={dynamic_spark_veto}, edge={edge_margin}px)")
    print(f"========================================================")

    tasks_to_run = []
    all_output_files = []

    for r in run_ranges:
        r_min = r["run_min"]
        r_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]
        r_ped = r.get("pedrun")

        filename = get_map_filename(r_min, r_max, vgem, vdrift, source=source, use_topoveto=use_topoveto)
        output_filename = os.path.join(output_dir, filename)
        all_output_files.append((r, output_filename))

        already_exists = os.path.exists(output_filename) and os.path.getsize(output_filename) > 0

        if already_exists and not overwrite:
            print(f"  > [SKIP - ESISTE GIÀ] Range {r_min}-{r_max} | VGEM={vgem}V, VD={vdrift}V -> {output_filename}")
            continue

        if source == "midas":
            cmd = [
                sys.executable, map_script,
                "--run-start", str(r_min),
                "--run-end", str(r_max),
                "--pedrun", str(r_ped),
                "--output", output_filename,
                "--nsigma", str(nsigma),
                "--tag", str(file_s3_tag),
                "--tmpdir", tmpdir,
                "--ped-dir", ped_dir,
                "--veto-events-dir", veto_events_dir,
                "--max-cache-gb", str(max_cache_gb),
                "--max-downloads", str(max_downloads),
                "--stall-timeout", str(stall_timeout),
                "--device", device,
                *build_midas_event_veto_flags(),
            ]
            if cleanup_midas:
                cmd.append("--cleanup-midas")
            if use_topoveto:
                cmd.extend(["--topo-veto-file", topo_veto_file])
            ped_info = f" | pedrun={r_ped}"
        else:
            cmd = [
                sys.executable, map_script,
                "--run-start", str(r_min),
                "--run-end", str(r_max),
                "--output", output_filename,
                "--workers", str(workers_per_run),
            ]
            ped_info = ""

        tasks_to_run.append((r_min, r_max, vgem, vdrift, cmd))
        print(f"  > [DA GENERARE]       Range {r_min}-{r_max} | VGEM={vgem}V, VD={vdrift}V{ped_info} -> {output_filename}")

    if not execute:
        print(f"\n(Modalità DRY-RUN: {len(tasks_to_run)} mappe da generare, {len(run_ranges) - len(tasks_to_run)} già presenti. Usa --execute per avviare)")
        return

    if source == "midas" and tasks_to_run:
        sys.path.insert(0, base_dir)
        from make_2dmaps_frommidas import get_device, get_or_create_pedestal

        needed_peds = sorted({r["pedrun"] for r, out_f in all_output_files if not (os.path.exists(out_f) and os.path.getsize(out_f) > 0) or overwrite})
        missing_peds = [p for p in needed_peds if not os.path.exists(os.path.join(ped_dir, f"pedestal_run{p:05d}.npz"))]
        if missing_peds:
            print(f"\n---> Pre-calcolo piedistalli mancanti: {missing_peds}")
            dev = get_device(device)
            for p in missing_peds:
                get_or_create_pedestal(
                    pedrun=p,
                    device=dev,
                    tag=file_s3_tag,
                    tmpdir=tmpdir,
                    ped_dir=ped_dir,
                    cleanup_midas=cleanup_midas,
                    max_cache_gb=max_cache_gb,
                    max_downloads=max_downloads,
                    stall_timeout=stall_timeout,
                )

    errors = []
    if tasks_to_run:
        max_cores = multiprocessing.cpu_count()
        if n_jobs <= 0 or n_jobs > max_cores:
            n_jobs = max_cores

        actual_workers = min(n_jobs, len(tasks_to_run))
        print(f"\n---> ESECUZIONE PARALLELA DI {len(tasks_to_run)} TASK SU {actual_workers} WORKER (max download S3 simultanei: {max_downloads})...\n")

        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            futures = [executor.submit(execute_command, task) for task in tasks_to_run]
            for future in as_completed(futures):
                ok, rmin, rmax, err = future.result()
                if not ok:
                    errors.append(err)
    else:
        print(f"\n---> Tutti i file ROOT esistono già: salto {os.path.basename(map_script)}.")

    if errors and error_log:
        with open(error_log, "a") as flog:
            for e in errors:
                flog.write(e + "\n")
        print(f"[WARNING] Registrati {len(errors)} errori in {error_log}")

    if plot_2dmaps and os.path.exists(plot_2d_script):
        print(f"\n---> Generazione plot 2D in: {plots_2d_dir}")
        for r, fpath in all_output_files:
            if not (os.path.exists(fpath) and os.path.getsize(fpath) > 0):
                continue
            out_name = f"run{r['run_min']}_{r['run_max']}" if r["run_min"] != r["run_max"] else f"run{r['run_min']}"
            p2d_cmd = [
                sys.executable, plot_2d_script,
                "--input", fpath,
                "--output-dir", plots_2d_dir,
                "--output-name", out_name,
            ]
            if source == "reco":
                p2d_cmd.append("--from-reco")
            subprocess.run(p2d_cmd, check=False)

    plot_cmd = [
        sys.executable, plot_script,
        "--csv", csv_file,
        "--source", source,
        "--file-s3-tag", file_s3_tag,
        "--mode", mode,
        "--output-dir", output_dir,
        "--summary-csv", summary_csv,
        "--plot-output", plot_output,
        "--time-unit", time_unit,
        "--edge-margin", str(edge_margin),
        "--spark-nsigma", str(spark_nsigma),
        "--spark-skirt-nsigma", str(spark_skirt_nsigma),
        "--spark-max-radius", str(spark_max_radius),
        "--spark-hard-cap", str(spark_hard_cap),
        "--spark-dilate", str(spark_dilate),
    ]
    if not dynamic_spark_veto:
        plot_cmd.append("--no-dynamic-spark-veto")
    if spark_ratio is not None:
        plot_cmd.extend(["--spark-ratio", str(spark_ratio)])
    if spark_abs is not None:
        plot_cmd.extend(["--spark-abs", str(spark_abs)])
    if spark_check_dir is not None:
        plot_cmd.extend(["--spark-check-dir", spark_check_dir])
    if not exclude_pedestals:
        plot_cmd.append("--include-pedestals")
    if group_by_desc:
        plot_cmd.append("--group-by-desc")
    if use_topoveto:
        plot_cmd.extend(["--topo-veto-file", topo_veto_file])
    if run_min is not None:
        plot_cmd.extend(["--run-min", str(run_min)])
    if run_max is not None:
        plot_cmd.extend(["--run-max", str(run_max)])
    if run_description is not None:
        if isinstance(run_description, (list, tuple)):
            plot_cmd.extend(["--run-description", *run_description])
        else:
            plot_cmd.extend(["--run-description", str(run_description)])
    if time_col is not None:
        plot_cmd.extend(["--time-col", time_col])

    print(f"\n---> Avvio analisi e plot: {' '.join(plot_cmd)}")
    subprocess.run(plot_cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa file CSV, crea le mappe 2D mancanti e lancia il plot vs tempo.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV di input")
    parser.add_argument("--source", type=str, choices=["midas", "reco"], default="midas", help="Sorgente dati: midas (raw images, default) o reco (clusters)")
    parser.add_argument("--pedrun", type=int, default=None, help="Run di piedistallo per modalità midas")
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
    parser.add_argument("--topo-veto-file", type=str, default=None, help="File .npz della maschera di veto topologico")
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

    parser.add_argument("--run-min", type=int, default=None, help="Numero del run minimo da includere")
    parser.add_argument("--run-max", type=int, default=None, help="Numero del run massimo da includere")
    parser.add_argument("--run-description", type=str, nargs="*", default=None, help="Valore/i di run_description da selezionare (default: nessun filtro)")
    parser.add_argument("--include-pedestals", action="store_true", help="Include anche i run di piedistallo se --run-description non è specificato")
    parser.add_argument("--group-by-desc", action="store_true", help="Separa le curve del grafico per run_description")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Valore di file_s3_tag da selezionare (default: LNGS)")
    parser.add_argument("--mode", type=str, choices=["per-run", "consecutive"], default="per-run", help="Modalità: per-run o consecutive")
    parser.add_argument("--execute", action="store_true", help="Esegue effettivamente i comandi e il plot")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Numero di job/processi paralleli (-1 per tutti i core)")
    parser.add_argument("-w", "--workers-per-run", type=int, default=4, help="Worker interni per singolo run in modalità reco (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="Rigenera i file ROOT anche se esistono già")
    parser.add_argument("--output-dir", type=str, default=".", help="Cartella in cui salvare/cercare i file ROOT (default: .)")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv", help="CSV di output con medie e tempi")
    parser.add_argument("--plot-output", type=str, default="trend_integral.png", help="File immagine di output per il plot")
    parser.add_argument("--plot-2dmaps", action="store_true", help="Genera anche i plot 2D (PNG/PDF) per ciascun run")
    parser.add_argument("--plots-2d-dir", type=str, default="plots_2d", help="Directory di output per le mappe 2D")
    parser.add_argument("--error-log", type=str, default="pipeline_errors.log", help="File di log errori")
    parser.add_argument("--time-col", type=str, default=None, help="Nome della colonna start-time nel CSV (opzionale)")
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours", help="Unità asse X (default: hours)")

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