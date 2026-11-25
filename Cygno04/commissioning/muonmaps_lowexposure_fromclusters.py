#!/usr/bin/env python3
import os
import sys
import glob
import time
import argparse
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

PED_REGEX = r"\bped(?:estal|s)?\b"


def get_map_filename(run_min, run_max, vgem, vdrift, source="midas", use_topoveto=False):
    veto_tag = "_topoveto" if use_topoveto else ""
    vgem_str = f"{int(vgem)}" if pd.notna(vgem) and float(vgem).is_integer() else f"{vgem}"
    vd_str = f"{int(vdrift)}" if pd.notna(vdrift) and float(vdrift).is_integer() else f"{vdrift}"
    if source == "midas":
        return f"cygno04_sideA_midas_raw{veto_tag}_VGEM_{vgem_str}_VD_{vd_str}_run_{run_min}_{run_max}.root"
    return f"cygno04_sideA_muons_withVignetting{veto_tag}_VGEM_{vgem_str}_VD_{vd_str}_run_{run_min}_{run_max}.root"


def find_existing_root(output_dir, expected_path, r0, r1):
    if os.path.exists(expected_path) and os.path.getsize(expected_path) > 0:
        return expected_path
    matches = sorted(glob.glob(os.path.join(output_dir, f"*_run_{r0}_{r1}.root")))
    matches = [
        m for m in matches
        if "_bare_" not in os.path.basename(m)
        and not m.endswith("_clean.root")
        and os.path.getsize(m) > 0
    ]
    return matches[0] if matches else None


def find_consecutive_run_ranges(df_filtered):
    df_s = df_filtered.sort_values("run_number").reset_index(drop=True)
    ranges = []
    current = None

    for _, row in df_s.iterrows():
        rnum = int(row["run_number"])
        vgem = row.get("GEM1_V", np.nan)
        vdrift = row.get("DRIFT_V", np.nan)
        ped = row.get("_resolved_pedrun", np.nan)

        if (
            current is not None
            and rnum == current["run_max"] + 1
            and np.isclose(vgem, current["vgem"], equal_nan=True)
            and np.isclose(vdrift, current["vdrift"], equal_nan=True)
            and np.isclose(ped, current["pedrun"], equal_nan=True)
        ):
            current["run_max"] = rnum
        else:
            if current is not None:
                ranges.append(current)
            current = {
                "run_min": rnum,
                "run_max": rnum,
                "vgem": vgem,
                "vdrift": vdrift,
                "pedrun": ped,
            }
    if current is not None:
        ranges.append(current)
    return ranges


def resolve_pedruns_from_csv(df_all, df_target, default_pedrun=None):
    df_target = df_target.copy()
    if default_pedrun is not None:
        df_target["_resolved_pedrun"] = int(default_pedrun)
        return df_target

    if "pedestal_run" in df_target.columns:
        ped_col = pd.to_numeric(df_target["pedestal_run"], errors="coerce")
        if ped_col.notna().all():
            df_target["_resolved_pedrun"] = ped_col.astype(int)
            return df_target

    is_ped = df_all["run_description"].astype(str).str.lower().str.contains(PED_REGEX, regex=True, na=False)
    ped_runs = sorted(df_all.loc[is_ped, "run_number"].dropna().astype(int).unique().tolist())
    if not ped_runs:
        raise ValueError("Nessun run di piedistallo trovato nel CSV e --pedrun non specificato.")

    ped_arr = np.array(ped_runs, dtype=int)
    resolved = []
    for rnum in df_target["run_number"].astype(int):
        prev_peds = ped_arr[ped_arr <= rnum]
        if len(prev_peds) > 0:
            resolved.append(int(prev_peds[-1]))
        else:
            resolved.append(int(ped_arr[0]))
    df_target["_resolved_pedrun"] = resolved
    return df_target


def enforce_cache_limit(tmpdir, max_cache_gb):
    if max_cache_gb is None or max_cache_gb <= 0:
        return
    files = glob.glob(os.path.join(tmpdir, "run*.mid.gz"))
    if not files:
        return
    entries = []
    total_bytes = 0
    for f in files:
        try:
            st = os.stat(f)
            entries.append((st.st_mtime, st.st_size, f))
            total_bytes += st.st_size
        except OSError:
            pass
    limit_bytes = max_cache_gb * (1024**3)
    entries.sort()
    for _, sz, fpath in entries:
        if total_bytes <= limit_bytes:
            break
        try:
            os.remove(fpath)
            total_bytes -= sz
        except OSError:
            pass


def run_with_watchdog(cmd, tmpdir, runs_in_job, stall_timeout=90, download_sem=None):
    if download_sem is not None:
        download_sem.acquire()
    try:
        proc = subprocess.Popen(cmd)
        last_sizes = {}
        last_progress = time.time()

        while proc.poll() is None:
            time.sleep(5)
            progressed = False
            for r in runs_in_job:
                f_mid = os.path.join(tmpdir, f"run{int(r):05d}.mid.gz")
                if os.path.exists(f_mid):
                    sz = os.path.getsize(f_mid)
                    if sz != last_sizes.get(r, -1):
                        last_sizes[r] = sz
                        progressed = True
            if progressed:
                last_progress = time.time()
            elif stall_timeout and (time.time() - last_progress) > stall_timeout:
                proc.kill()
                proc.wait()
                for r in runs_in_job:
                    f_mid = os.path.join(tmpdir, f"run{int(r):05d}.mid.gz")
                    if os.path.exists(f_mid):
                        try:
                            os.remove(f_mid)
                        except OSError:
                            pass
                return -9, f"Timeout stallo download (> {stall_timeout}s)"
        return proc.returncode, ""
    finally:
        if download_sem is not None:
            download_sem.release()


def run_map_processing(
    csv_file="commissioning/runlist_stopped_muons.csv",
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
    z_min=0.0,
    z_max="auto",
    error_log="pipeline_errors.log",
    time_col=None,
    time_unit="hours",
):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    midas_script = os.path.join(repo_root, "make_2dmaps_frommidas.py")
    reco_script = os.path.join(repo_root, "make_2dmaps_fromclusters_s3.py")
    veto_script = os.path.join(repo_root, "find_gem_spark_veto.py")
    plot_script = os.path.join(repo_root, "plot_xy_mean_vs_time.py")

    df = pd.read_csv(csv_file)
    df["run_number"] = pd.to_numeric(df["run_number"], errors="coerce")
    df = df.dropna(subset=["run_number"]).copy()
    df["run_number"] = df["run_number"].astype(int)

    if "file_s3_tag" in df.columns and file_s3_tag:
        df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()
        df_tag = df[df["file_s3_tag"] == file_s3_tag].copy()
    else:
        df_tag = df.copy()

    df_filt = df_tag.copy()
    if "run_description" in df_filt.columns:
        df_filt["run_description"] = df_filt["run_description"].astype(str).str.strip()
        if run_description is not None:
            desc_list = [run_description] if isinstance(run_description, str) else list(run_description)
            df_filt = df_filt[df_filt["run_description"].isin(desc_list)].copy()
        elif exclude_pedestals:
            is_ped = df_filt["run_description"].str.lower().str.contains(PED_REGEX, regex=True, na=False)
            if "pedestal_run" in df_filt.columns:
                ped_col = pd.to_numeric(df_filt["pedestal_run"], errors="coerce")
                is_ped |= (df_filt["run_number"] == ped_col)
            df_filt = df_filt[~is_ped].copy()

    if run_min is not None:
        df_filt = df_filt[df_filt["run_number"] >= int(run_min)].copy()
    if run_max is not None:
        df_filt = df_filt[df_filt["run_number"] <= int(run_max)].copy()

    if df_filt.empty:
        print("Nessun run trovato con i filtri specificati.")
        return

    for col in ("GEM1_V", "DRIFT_V"):
        if col in df_filt.columns:
            df_filt[col] = pd.to_numeric(df_filt[col], errors="coerce")

    if source == "midas":
        df_filt = resolve_pedruns_from_csv(df_tag, df_filt, default_pedrun=pedrun)
    else:
        df_filt["_resolved_pedrun"] = np.nan

    # STEP 1: Creazione veto topologico da un run bare di riferimento
    if make_topo_veto:
        ref_run = int(veto_ref_run) if veto_ref_run is not None else int(df_filt["run_number"].min())
        ref_rows = df_filt[df_filt["run_number"] == ref_run]
        if ref_rows.empty:
            ref_rows = df_tag[df_tag["run_number"] == ref_run]
        if ref_rows.empty:
            raise ValueError(f"Run di riferimento {ref_run} non trovato nel CSV.")
        ref_row = ref_rows.iloc[0]
        ref_ped = int(ref_row["_resolved_pedrun"]) if pd.notna(ref_row.get("_resolved_pedrun", np.nan)) else int(pedrun)
        vgem = ref_row.get("GEM1_V", np.nan)
        vdrift = ref_row.get("DRIFT_V", np.nan)
        vgem_str = f"{int(vgem)}" if pd.notna(vgem) and float(vgem).is_integer() else f"{vgem}"
        vd_str = f"{int(vdrift)}" if pd.notna(vdrift) and float(vdrift).is_integer() else f"{vdrift}"

        bare_root = os.path.join(output_dir, f"cygno04_sideA_midas_raw_bare_VGEM_{vgem_str}_VD_{vd_str}_run_{ref_run}_{ref_run}.root")
        veto_out_npz = topo_veto_file if topo_veto_file is not None else "gem_spark_veto.npz"

        print(f"\n[STEP 1 — TOPO VETO] Run riferimento: {ref_run} (pedrun={ref_ped})")
        print(f"  Mappa bare: {bare_root}")
        print(f"  Veto output: {veto_out_npz}")

        if not execute:
            print("  (Dry-run: aggiungi --execute per eseguire lo Step 1)")
            return

        if overwrite or not os.path.exists(bare_root):
            cmd_bare = [
                sys.executable, midas_script,
                "--run-start", str(ref_run),
                "--run-end", str(ref_run),
                "--pedrun", str(ref_ped),
                "--output", bare_root,
                "--nsigma", str(nsigma),
                "--tag", file_s3_tag,
                "--tmpdir", tmpdir,
                "--ped-dir", ped_dir,
                "--veto-events-dir", veto_events_dir,
                "--no-event-spark-veto",
                "--device", device,
            ]
            subprocess.run(cmd_bare, check=True)
            if cleanup_midas:
                f_mid = os.path.join(tmpdir, f"run{ref_run:05d}.mid.gz")
                if os.path.exists(f_mid):
                    os.remove(f_mid)

        cmd_veto = [
            sys.executable, veto_script,
            "--input", bare_root,
            "--source", source,
            "--veto-output", veto_out_npz,
            "--plot-output", veto_plot_output,
            "--nsigma-spark", str(topo_spark_nsigma),
            "--edge-margin", str(edge_margin),
            "--dilate", str(spark_dilate),
        ]
        subprocess.run(cmd_veto, check=True)
        print(f"\nStep 1 completato. Controlla '{veto_plot_output}' e poi lancia lo Step 2 con:")
        print(f"  python commissioning/run_orchestrator.py --csv {csv_file} --topo-veto-file {veto_out_npz} --plot-2dmaps --execute -j 2\n")
        return

    # STEP 2: Generazione mappe 2D per tutti i blocchi / run
    blocks = [
        {
            "run_min": int(r["run_number"]),
            "run_max": int(r["run_number"]),
            "vgem": r.get("GEM1_V", np.nan),
            "vdrift": r.get("DRIFT_V", np.nan),
            "pedrun": r.get("_resolved_pedrun", np.nan),
        }
        for _, r in df_filt.sort_values("run_number").iterrows()
    ] if mode == "per-run" else find_consecutive_run_ranges(df_filt)

    use_topoveto = topo_veto_file is not None
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    for blk in blocks:
        r0, r1 = blk["run_min"], blk["run_max"]
        vgem, vdrift = blk["vgem"], blk["vdrift"]
        fname = get_map_filename(r0, r1, vgem, vdrift, source=source, use_topoveto=use_topoveto)
        out_root = os.path.join(output_dir, fname)

        if source == "midas":
            cmd = [
                sys.executable, midas_script,
                "--run-start", str(r0),
                "--run-end", str(r1),
                "--pedrun", str(int(blk["pedrun"])),
                "--output", out_root,
                "--nsigma", str(nsigma),
                "--tag", file_s3_tag,
                "--tmpdir", tmpdir,
                "--ped-dir", ped_dir,
                "--veto-events-dir", veto_events_dir,
                "--edge-margin", str(edge_margin),
                "--device", device,
            ]
            if topo_veto_file is not None:
                cmd.extend(["--topo-veto-file", topo_veto_file])
            if not event_spark_veto:
                cmd.append("--no-event-spark-veto")
            else:
                cmd.extend([
                    "--event-spark-nsigma", str(event_spark_nsigma),
                    "--event-spark-skirt-nsigma", str(event_spark_skirt_nsigma),
                    "--event-spark-max-radius", str(event_spark_max_radius),
                    "--event-spark-hard-cap", str(event_spark_hard_cap),
                    "--event-spark-dilate", str(event_spark_dilate),
                ])
        else:
            cmd = [
                sys.executable, reco_script,
                "--run-start", str(r0),
                "--run-end", str(r1),
                "--output", out_root,
                "--workers", str(workers_per_run),
            ]

        existing = None if overwrite else find_existing_root(output_dir, out_root, r0, r1)
        tasks.append((blk, out_root, cmd, existing))

    to_run = [t for t in tasks if t[3] is None]
    print(f"\n[ORCHESTRATOR] Trovati {len(tasks)} blocchi/run totali ({len(to_run)} da elaborare, {len(tasks) - len(to_run)} già presenti).")

    if not execute:
        for blk, out_root, _, _ in to_run[:15]:
            print(f"  [DRY-RUN] Run {blk['run_min']}..{blk['run_max']} (ped={blk['pedrun']}) -> {os.path.basename(out_root)}")
        if len(to_run) > 15:
            print(f"  ... e altri {len(to_run) - 15} run.")
        print("\nAggiungi --execute per avviare l'elaborazione.")
        return

    # Pre-calcolo piedistalli unici prima del parallelismo (solo se ci sono run da elaborare)
    if source == "midas" and to_run:
        unique_peds = sorted({int(t[0]["pedrun"]) for t in to_run if pd.notna(t[0]["pedrun"])})
        for uped in unique_peds:
            ped_cache = os.path.join(ped_dir, f"pedestal_run{uped:05d}.npz")
            if not os.path.exists(ped_cache):
                print(f"[PRE-PEDESTAL] Generazione cache piedistallo run {uped:05d}...")
                pre_cmd = [
                    sys.executable, "-c",
                    (
                        f"import sys; sys.path.insert(0, {repo_root!r}); "
                        f"from make_2dmaps_frommidas import get_device, get_or_create_pedestal; "
                        f"dev = get_device({device!r}); "
                        f"get_or_create_pedestal({uped}, dev, tag={file_s3_tag!r}, tmpdir={tmpdir!r}, ped_dir={ped_dir!r})"
                    ),
                ]
                subprocess.run(pre_cmd, check=True)
                if cleanup_midas:
                    f_ped_mid = os.path.join(tmpdir, f"run{uped:05d}.mid.gz")
                    if os.path.exists(f_ped_mid):
                        os.remove(f_ped_mid)

    download_sem = threading.Semaphore(max(1, int(max_downloads)))
    errors = []

    def execute_task(item):
        blk, out_root, cmd, _ = item
        runs_in_job = list(range(blk["run_min"], blk["run_max"] + 1))
        for attempt in range(1, 3):
            if not cleanup_midas:
                enforce_cache_limit(tmpdir, max_cache_gb)
            rc, msg = run_with_watchdog(
                cmd=cmd,
                tmpdir=tmpdir,
                runs_in_job=runs_in_job,
                stall_timeout=stall_timeout,
                download_sem=download_sem,
            )
            if rc == 0 and os.path.exists(out_root):
                if cleanup_midas:
                    for r in runs_in_job:
                        f_mid = os.path.join(tmpdir, f"run{int(r):05d}.mid.gz")
                        if os.path.exists(f_mid):
                            try:
                                os.remove(f_mid)
                            except OSError:
                                pass
                return True, f"Run {blk['run_min']}..{blk['run_max']} OK"
            print(f"[RETRY {attempt}/2] Run {blk['run_min']}..{blk['run_max']} fallito (rc={rc} {msg})")
        return False, f"Run {blk['run_min']}..{blk['run_max']} FALLITO"

    if to_run:
        with ThreadPoolExecutor(max_workers=max(1, int(n_jobs))) as pool:
            futures = {pool.submit(execute_task, item): item for item in to_run}
            for fut in as_completed(futures):
                ok, status_msg = fut.result()
                print(f"  -> {status_msg}")
                if not ok:
                    errors.append(status_msg)

    if errors:
        with open(error_log, "w") as flog:
            for e in errors:
                flog.write(e + "\n")
        print(f"[ATTENZIONE] {len(errors)} job falliti. Dettagli in: {error_log}")

    # Lancio finale di plot_xy_mean_vs_time.py
    effective_check_dir = spark_check_dir
    if effective_check_dir is None and plot_2dmaps and dynamic_spark_veto:
        effective_check_dir = os.path.join(plots_2d_dir, "spark_checks")

    plot_cmd = [
        sys.executable, plot_script,
        "--csv", csv_file,
        "--source", source,
        "--file-s3-tag", file_s3_tag,
        "--mode", mode,
        "--output-dir", output_dir,
        "--summary-csv", summary_csv,
        "--plot-output", plot_output,
        "--plots-2d-dir", plots_2d_dir,
        "--z-min", str(z_min),
        "--z-max", str(z_max),
        "--edge-margin", str(edge_margin),
        "--spark-nsigma", str(spark_nsigma),
        "--spark-skirt-nsigma", str(spark_skirt_nsigma),
        "--spark-max-radius", str(spark_max_radius),
        "--spark-hard-cap", str(spark_hard_cap),
        "--spark-dilate", str(spark_dilate),
        "--time-unit", time_unit,
    ]
    if run_min is not None:
        plot_cmd.extend(["--run-min", str(run_min)])
    if run_max is not None:
        plot_cmd.extend(["--run-max", str(run_max)])
    if run_description is not None:
        desc_list = [run_description] if isinstance(run_description, str) else list(run_description)
        plot_cmd.extend(["--run-description", *desc_list])
    if not exclude_pedestals:
        plot_cmd.append("--include-pedestals")
    if group_by_desc:
        plot_cmd.append("--group-by-desc")
    if topo_veto_file is not None:
        plot_cmd.extend(["--topo-veto-file", topo_veto_file])
    if not dynamic_spark_veto:
        plot_cmd.append("--no-dynamic-spark-veto")
    if spark_ratio is not None:
        plot_cmd.extend(["--spark-ratio", str(spark_ratio)])
    if spark_abs is not None:
        plot_cmd.extend(["--spark-abs", str(spark_abs)])
    if plot_2dmaps:
        plot_cmd.append("--plot-2dmaps")
    if effective_check_dir is not None:
        plot_cmd.extend(["--spark-check-dir", effective_check_dir])
    if time_col is not None:
        plot_cmd.extend(["--time-col", time_col])

    subprocess.run(plot_cmd, check=True)