import os
import argparse
from collections import deque
import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PEDESTAL_REGEX = r"(?:^|[\s_\-])ped(?:estal)?(?:$|[\s_\-])"


def get_map_filename(run_min, run_max, vgem, vdrift, source="midas", use_topoveto=False):
    veto_tag = "_topoveto" if use_topoveto else ""
    if source == "midas":
        return f"cygno04_sideA_midas_raw{veto_tag}_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"
    return f"cygno04_sideA_muons_withVignetting{veto_tag}_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"


def detect_time_column(df, user_col=None):
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(f"Colonna '{user_col}' non trovata nel CSV. Colonne disponibili: {list(df.columns)}")
        return user_col

    candidates = [
        "start_time", "start-time", "starttime", "Start_Time", "Start-Time",
        "time_start", "t_start", "timestamp", "time", "date"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    raise ValueError(
        f"Impossibile individuare automaticamente la colonna start-time. "
        f"Specificala con --time-col. Colonne disponibili: {list(df.columns)}"
    )


def parse_timestamps(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum() and numeric.notna().any():
        median_val = numeric.dropna().median()
        unit = "ms" if median_val > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def binary_dilate_2d(mask, radius=1):
    if radius <= 0 or not np.any(mask):
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


def find_connected_components(mask):
    """Restituisce la lista dei componenti connessi (8-connettività) come liste di tuple (ix, iy)."""
    nx, ny = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for ix, iy in zip(*np.where(mask)):
        if visited[ix, iy]:
            continue
        comp = []
        queue = deque([(int(ix), int(iy))])
        visited[ix, iy] = True
        while queue:
            cx, cy = queue.popleft()
            comp.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx_i, ny_i = cx + dx, cy + dy
                    if 0 <= nx_i < nx and 0 <= ny_i < ny:
                        if mask[nx_i, ny_i] and not visited[nx_i, ny_i]:
                            visited[nx_i, ny_i] = True
                            queue.append((nx_i, ny_i))
        components.append(comp)
    return components


def detect_bounded_spark_mask(
    prof,
    base_valid,
    core_thr,
    base_skirt_thr,
    peak_fraction=0.02,
    max_grow_radius=4,
    dilate_bins=1,
):
    """
    Individua le scariche sopra core_thr ed espande ciascuna componente solo entro
    max_grow_radius bin dal nucleo usando una soglia adattiva al picco locale,
    evitando di invadere il fondo diffuso normale del rivelatore.
    """
    core_mask = base_valid & (prof > core_thr)
    if not np.any(core_mask):
        return np.zeros_like(base_valid, dtype=bool)

    nx, ny = prof.shape
    grown_union = np.zeros_like(base_valid, dtype=bool)
    components = find_connected_components(core_mask)

    for comp in components:
        comp_mask = np.zeros_like(base_valid, dtype=bool)
        for ix, iy in comp:
            comp_mask[ix, iy] = True

        comp_peak = float(np.max(prof[comp_mask]))
        local_skirt_thr = min(float(core_thr), max(float(base_skirt_thr), float(peak_fraction) * comp_peak))

        if max_grow_radius is not None and max_grow_radius > 0:
            allowed_zone = binary_dilate_2d(comp_mask, radius=max_grow_radius)
        else:
            allowed_zone = np.ones_like(base_valid, dtype=bool)

        candidates = base_valid & allowed_zone & (prof > local_skirt_thr)
        comp_grown = comp_mask.copy()
        queue = deque(comp)

        while queue:
            cx, cy = queue.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx_i, ny_i = cx + dx, cy + dy
                    if 0 <= nx_i < nx and 0 <= ny_i < ny:
                        if candidates[nx_i, ny_i] and not comp_grown[nx_i, ny_i]:
                            comp_grown[nx_i, ny_i] = True
                            queue.append((nx_i, ny_i))

        grown_union |= comp_grown

    final_mask = binary_dilate_2d(grown_union, radius=dilate_bins) & base_valid
    return final_mask


def save_dynamic_spark_check(
    run_label,
    prof,
    x_edges,
    y_edges,
    base_valid,
    dynamic_veto_mask,
    clean_valid,
    threshold,
    check_dir,
):
    os.makedirs(check_dir, exist_ok=True)
    out_png = os.path.join(check_dir, f"spark_clean_{run_label}.png")

    prof_before = np.where(base_valid, prof, np.nan).T
    prof_after = np.where(clean_valid, prof, np.nan).T
    veto_overlay = np.where(dynamic_veto_mask, 1.0, np.nan).T

    max_before = float(np.nanmax(prof_before)) if np.any(base_valid) else 0.0
    max_after = float(np.nanmax(prof_after)) if np.any(clean_valid) else 0.0
    n_removed = int(np.sum(dynamic_veto_mask))

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), dpi=150)

    m0 = axes[0].pcolormesh(x_edges, y_edges, prof_before, shading="flat", cmap="rainbow")
    cb0 = fig.colorbar(m0, ax=axes[0], pad=0.02, fraction=0.046)
    cb0.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[0].set_title(f"1. Prima (max={max_before:.0f} ADC)", fontsize=11, fontweight="bold")

    axes[1].pcolormesh(x_edges, y_edges, prof_after, shading="flat", cmap="Greys", alpha=0.55)
    axes[1].pcolormesh(x_edges, y_edges, veto_overlay, shading="flat", cmap="autumn", vmin=0, vmax=1, alpha=0.95)
    axes[1].set_title(f"2. Spike rimossa ({n_removed} bin | thr={threshold:.1f})", fontsize=11, fontweight="bold")

    m2 = axes[2].pcolormesh(x_edges, y_edges, prof_after, shading="flat", cmap="rainbow")
    cb2 = fig.colorbar(m2, ax=axes[2], pad=0.02, fraction=0.046)
    cb2.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[2].set_title(f"3. Dopo pulizia (max={max_after:.0f} ADC)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("X [pixel]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    axes[0].set_ylabel("Y [pixel]", fontsize=10)

    fig.suptitle(f"Rimozione dinamica spike — {run_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def find_consecutive_run_ranges_with_time(df, split_by_desc=False):
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)

    ranges = []
    current_start = df.loc[0, "run_number"]
    current_end = df.loc[0, "run_number"]
    current_vgem = df.loc[0, "GEM1_V"]
    current_vdrift = df.loc[0, "DRIFT_V"]
    first_desc = str(df.loc[0, "run_description"]) if "run_description" in df.columns else ""
    current_descs = [first_desc] if first_desc else []
    current_t_start = df.loc[0, "_parsed_time"]
    current_t_end = df.loc[0, "_parsed_time"]

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]
        desc = str(row["run_description"]) if "run_description" in df.columns else ""
        t_run = row["_parsed_time"]

        same_desc = (desc == current_descs[-1]) if (split_by_desc and current_descs) else True

        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift) and same_desc:
            current_end = run_num
            if desc and desc not in current_descs:
                current_descs.append(desc)
            if pd.notnull(t_run):
                current_t_end = t_run
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
                "run_description": " | ".join(current_descs),
                "t_start": current_t_start,
                "t_end": current_t_end,
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift
            current_descs = [desc] if desc else []
            current_t_start = t_run
            current_t_end = t_run

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
        "run_description": " | ".join(current_descs),
        "t_start": current_t_start,
        "t_end": current_t_end,
    })

    return ranges


def get_per_run_ranges_with_time(df):
    if df.empty:
        return []

    df = df.sort_values("run_number").reset_index(drop=True)
    ranges = []
    for _, row in df.iterrows():
        run_num = int(row["run_number"])
        vgem = int(row["GEM1_V"]) if pd.notnull(row["GEM1_V"]) else row["GEM1_V"]
        vdrift = int(row["DRIFT_V"]) if pd.notnull(row["DRIFT_V"]) else row["DRIFT_V"]
        desc = str(row["run_description"]) if "run_description" in df.columns else ""
        t_run = row["_parsed_time"]
        ranges.append({
            "run_min": run_num,
            "run_max": run_num,
            "vgem": vgem,
            "vdrift": vdrift,
            "run_description": desc,
            "t_start": t_run,
            "t_end": t_run,
        })
    return ranges


def compute_xy_mean_from_root(
    root_path,
    source="midas",
    topo_veto_mask=None,
    edge_margin=100.0,
    dynamic_spark_veto=True,
    spark_nsigma=20.0,
    spark_skirt_nsigma=12.0,
    spark_max_radius=4,
    spark_hard_cap=150.0,
    spark_ratio=None,
    spark_abs=None,
    spark_dilate=1,
    spark_check_dir=None,
    run_label=None,
):
    with uproot.open(root_path) as f:
        if source == "midas":
            prof_key = "prof2d_images" if "prof2d_images" in f else "prof2d"
            occ_key = "occ2d_images" if "occ2d_images" in f else "occ2d"
        else:
            prof_key = "prof2d"
            occ_key = "occ2d"

        prof, x_edges, y_edges = f[prof_key].to_numpy()
        occ = f[occ_key].values()

    valid = (occ > 0) & np.isfinite(prof)

    if edge_margin is not None and edge_margin > 0:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_in = (x_centers >= (x_edges[0] + edge_margin)) & (x_centers <= (x_edges[-1] - edge_margin))
        y_in = (y_centers >= (y_edges[0] + edge_margin)) & (y_centers <= (y_edges[-1] - edge_margin))
        valid &= np.outer(x_in, y_in)

    if topo_veto_mask is not None:
        if topo_veto_mask.shape != valid.shape:
            raise ValueError(f"Shape veto_mask {topo_veto_mask.shape} incompatibile con mappa {valid.shape}")
        valid &= (~topo_veto_mask)

    base_valid = valid.copy()
    max_before = float(np.max(prof[base_valid])) if np.any(base_valid) else np.nan
    n_dyn_vetoed = 0

    if dynamic_spark_veto and np.any(base_valid):
        prof_active = prof[base_valid]
        median_val = float(np.median(prof_active))
        mad_val = float(np.median(np.abs(prof_active - median_val)))
        robust_std = 1.4826 * mad_val
        if robust_std <= 0:
            q75, q25 = np.percentile(prof_active, [75, 25])
            robust_std = float((q75 - q25) / 1.349) if q75 > q25 else float(np.std(prof_active))

        if spark_abs is not None:
            core_thr = float(spark_abs)
            base_skirt_thr = median_val + 0.6 * (core_thr - median_val)
        elif spark_ratio is not None:
            core_thr = float(spark_ratio) * median_val
            base_skirt_thr = median_val + 0.6 * (core_thr - median_val)
        else:
            core_thr = median_val + float(spark_nsigma) * robust_std
            base_skirt_thr = median_val + float(spark_skirt_nsigma) * robust_std

        has_hard_cap = spark_hard_cap is not None and float(spark_hard_cap) > 0
        if has_hard_cap:
            core_thr = min(core_thr, float(spark_hard_cap))
            base_skirt_thr = min(base_skirt_thr, core_thr)

        dyn_mask = detect_bounded_spark_mask(
            prof=prof,
            base_valid=base_valid,
            core_thr=core_thr,
            base_skirt_thr=base_skirt_thr,
            peak_fraction=0.02,
            max_grow_radius=spark_max_radius,
            dilate_bins=spark_dilate,
        )

        # Rete di sicurezza assoluta per singoli pixel/micro-spark sopra spark_hard_cap
        if has_hard_cap:
            hard_cap_hits = base_valid & (prof > float(spark_hard_cap))
            if np.any(hard_cap_hits):
                dyn_mask |= binary_dilate_2d(hard_cap_hits, radius=spark_dilate) & base_valid

        if np.any(dyn_mask):
            n_dyn_vetoed = int(np.sum(dyn_mask))
            valid &= (~dyn_mask)

            label = run_label or os.path.splitext(os.path.basename(root_path))[0]
            max_after_tmp = float(np.max(prof[valid])) if np.any(valid) else np.nan
            cap_str = f", cap={float(spark_hard_cap):.0f}" if has_hard_cap else ""
            print(
                f"  [DYNAMIC SPARK VETO] {label}: rimossi {n_dyn_vetoed} bin "
                f"(thr={core_thr:.1f}{cap_str} ADC | max: {max_before:.1f} -> {max_after_tmp:.1f} ADC)"
            )
            if spark_check_dir:
                save_dynamic_spark_check(
                    run_label=label,
                    prof=prof,
                    x_edges=x_edges,
                    y_edges=y_edges,
                    base_valid=base_valid,
                    dynamic_veto_mask=dyn_mask,
                    clean_valid=valid,
                    threshold=core_thr,
                    check_dir=spark_check_dir,
                )

    n_active = int(np.sum(valid))

    if n_active == 0:
        return {
            "mean_xy": np.nan,
            "std_xy": np.nan,
            "sem_xy": np.nan,
            "integral_xy": 0.0,
            "active_bins": 0,
            "dynamic_veto_bins": n_dyn_vetoed,
            "max_before_clean": max_before,
            "max_after_clean": np.nan,
        }

    prof_valid = prof[valid]
    mean_xy = float(np.mean(prof_valid))
    std_xy = float(np.std(prof_valid, ddof=1)) if n_active > 1 else 0.0
    sem_xy = float(std_xy / np.sqrt(n_active))
    integral_xy = float(np.sum(prof_valid))
    max_after = float(np.max(prof_valid))

    return {
        "mean_xy": mean_xy,
        "std_xy": std_xy,
        "sem_xy": sem_xy,
        "integral_xy": integral_xy,
        "active_bins": n_active,
        "dynamic_veto_bins": n_dyn_vetoed,
        "max_before_clean": max_before,
        "max_after_clean": max_after,
    }


def analyze_and_plot(
    csv_file,
    source="midas",
    topo_veto_file=None,
    edge_margin=100.0,
    dynamic_spark_veto=True,
    spark_nsigma=20.0,
    spark_skirt_nsigma=12.0,
    spark_max_radius=4,
    spark_hard_cap=150.0,
    spark_ratio=None,
    spark_abs=None,
    spark_dilate=1,
    spark_check_dir=None,
    run_description=None,
    exclude_pedestals=True,
    group_by_desc=False,
    run_min=None,
    run_max=None,
    file_s3_tag="LNGS",
    mode="per-run",
    output_dir=".",
    summary_csv="xy_mean_summary.csv",
    plot_output="trend_integral.png",
    time_col=None,
    time_unit="hours",
):
    print(f"\n========================================================")
    print(f"ANALISI TEMPORALE E PLOT MEDIA IN (X, Y) [source={source}]")
    print(f"========================================================")

    if edge_margin is not None and edge_margin > 0:
        print(f"--> Esclusione bordi esterni attiva: {edge_margin:.0f} pixel per lato")

    topo_veto_mask = None
    use_topoveto = topo_veto_file is not None
    if use_topoveto:
        if not os.path.exists(topo_veto_file):
            raise FileNotFoundError(f"File veto topologico non trovato: {topo_veto_file}")
        topo_veto_mask = np.load(topo_veto_file)["veto_mask"].astype(bool)
        print(f"--> Veto topologico statico attivo: {topo_veto_file} ({int(np.sum(topo_veto_mask))} bin esclusi)")

    if dynamic_spark_veto:
        cap_msg = f", hard_cap={spark_hard_cap:.0f} ADC" if (spark_hard_cap is not None and spark_hard_cap > 0) else ""
        print(
            f"--> Rimozione dinamica spike su prof2d ATTIVA "
            f"(nsigma={spark_nsigma}, skirt={spark_skirt_nsigma}, max_radius={spark_max_radius}, dilate={spark_dilate}{cap_msg})"
        )

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
        mask &= (~is_ped)

    df_filtered = df[mask].copy().sort_values("run_number").reset_index(drop=True)

    if df_filtered.empty:
        print(f"[WARNING] Nessun run trovato nel CSV con i criteri specificati.")
        return

    detected_col = detect_time_column(df_filtered, user_col=time_col)
    print(f"--> Colonna temporale utilizzata: '{detected_col}'")
    df_filtered["_parsed_time"] = parse_timestamps(df_filtered[detected_col])

    if df_filtered["_parsed_time"].isna().all():
        raise ValueError(f"Impossibile interpretare i valori della colonna '{detected_col}' come timestamp.")

    t0 = df_filtered["_parsed_time"].min()
    print(f"--> Tempo t = 0 (primo run): {t0}")

    unit_factors = {
        "seconds": (1.0, "Tempo dal primo run [s]"),
        "hours": (3600.0, "Tempo dal primo run [ore]"),
        "days": (86400.0, "Tempo dal primo run [giorni]"),
    }
    scale, xlabel = unit_factors[time_unit]

    if mode == "consecutive":
        run_ranges = find_consecutive_run_ranges_with_time(df_filtered, split_by_desc=group_by_desc)
    else:
        run_ranges = get_per_run_ranges_with_time(df_filtered)

    rows = []
    for r in run_ranges:
        r_min = r["run_min"]
        r_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]
        desc = r.get("run_description", "")

        fname = get_map_filename(r_min, r_max, vgem, vdrift, source=source, use_topoveto=use_topoveto)
        fpath = os.path.join(output_dir, fname)

        if not (os.path.exists(fpath) and os.path.getsize(fpath) > 0):
            print(f"[WARNING] File ROOT mancante o vuoto, saltato: {fpath}")
            continue

        run_label = f"run{r_min}_{r_max}" if r_min != r_max else f"run{r_min}"
        try:
            stats = compute_xy_mean_from_root(
                fpath,
                source=source,
                topo_veto_mask=topo_veto_mask,
                edge_margin=edge_margin,
                dynamic_spark_veto=dynamic_spark_veto,
                spark_nsigma=spark_nsigma,
                spark_skirt_nsigma=spark_skirt_nsigma,
                spark_max_radius=spark_max_radius,
                spark_hard_cap=spark_hard_cap,
                spark_ratio=spark_ratio,
                spark_abs=spark_abs,
                spark_dilate=spark_dilate,
                spark_check_dir=spark_check_dir,
                run_label=run_label,
            )
        except Exception as e:
            print(f"[ERRORE] Impossibile leggere {fpath}: {e}")
            continue

        dt_start = (r["t_start"] - t0).total_seconds() / scale
        dt_end = (r["t_end"] - t0).total_seconds() / scale
        dt_mid = 0.5 * (dt_start + dt_end)

        rows.append({
            "run_min": r_min,
            "run_max": r_max,
            "run_description": desc,
            "VGEM": vgem,
            "DRIFT_V": vdrift,
            "t_start_utc": r["t_start"].isoformat() if pd.notnull(r["t_start"]) else "",
            "t_end_utc": r["t_end"].isoformat() if pd.notnull(r["t_end"]) else "",
            f"dt_start_{time_unit}": dt_start,
            f"dt_mid_{time_unit}": dt_mid,
            f"dt_end_{time_unit}": dt_end,
            "file": fname,
            **stats,
        })

    if not rows:
        print("[WARNING] Nessun dato valido disponibile per il summary o il plot.")
        return

    out_df = pd.DataFrame(rows).sort_values(f"dt_start_{time_unit}").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    out_df.to_csv(summary_csv, index=False)

    print("\nRiepilogo calcolato:")
    cols_print = [
        "run_min", "run_max", "VGEM", "DRIFT_V",
        f"dt_mid_{time_unit}", "mean_xy", "sem_xy",
        "dynamic_veto_bins", "max_before_clean", "max_after_clean"
    ]
    cols_to_show = [c for c in cols_print if c in out_df.columns]
    print(out_df[cols_to_show].to_string(index=False))
    print(f"\n--> Tabella salvata in: {summary_csv}")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    colors = ["black", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "<"]

    group_cols = ["run_description", "VGEM", "DRIFT_V"] if group_by_desc else ["VGEM", "DRIFT_V"]
    grouped = out_df.groupby(group_cols, sort=False)

    for idx, (keys, grp) in enumerate(grouped):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        if group_by_desc:
            desc, vgem, vdrift = keys
            label = f"{desc} (VGEM={vgem}V, VD={vdrift}V)"
        else:
            vgem, vdrift = keys
            label = f"VGEM={vgem}V, VD={vdrift}V"

        x = grp[f"dt_mid_{time_unit}"].to_numpy()
        xerr_low = x - grp[f"dt_start_{time_unit}"].to_numpy()
        xerr_high = grp[f"dt_end_{time_unit}"].to_numpy() - x
        y_mean = grp["mean_xy"].to_numpy()
        y_err = grp["sem_xy"].to_numpy()

        ax.errorbar(
            x,
            y_mean,
            xerr=[xerr_low, xerr_high] if mode == "consecutive" else None,
            yerr=y_err,
            fmt=f"{marker}-",
            color=color,
            capsize=3,
            markersize=6,
            linewidth=1.3,
            label=label,
        )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Light integral (ADC/pixel)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(frameon=True, fontsize=10, loc="best")

    # Scala Y leggermente piu' grande (~12% di escursione aggiuntiva rispetto ai limiti automatici)
    cur_ymin, cur_ymax = ax.get_ylim()
    cur_span = cur_ymax - cur_ymin
    if cur_span > 0:
        ax.set_ylim(cur_ymin - 0.04 * cur_span, cur_ymax + 0.08 * cur_span)

    # Secondo asse X superiore con i numeri di run (senza titolo del grafico)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)

    tick_df = (
        out_df[["run_min", "run_max", f"dt_mid_{time_unit}"]]
        .drop_duplicates()
        .sort_values(f"dt_mid_{time_unit}")
        .reset_index(drop=True)
    )
    if len(tick_df) > 30:
        step = int(np.ceil(len(tick_df) / 25.0))
        tick_df = tick_df.iloc[::step].reset_index(drop=True)

    tick_positions = tick_df[f"dt_mid_{time_unit}"].to_numpy()
    tick_labels = [
        str(int(row["run_min"])) if row["run_min"] == row["run_max"] else f"{int(row['run_min'])}-{int(row['run_max'])}"
        for _, row in tick_df.iterrows()
    ]

    ax_top = ax.twiny()
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(tick_positions)
    ax_top.set_xticklabels(tick_labels, fontsize=7.5, rotation=45, ha="left")
    ax_top.tick_params(axis="x", which="major", pad=2, length=4)
    ax_top.set_xlabel("Run number", fontsize=10, labelpad=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_output)), exist_ok=True)
    plt.savefig(plot_output)
    plt.close(fig)
    print(f"--> Plot salvato in: {plot_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola la media e l'integrale delle mappe CYGNO e genera il plot vs tempo.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV iniziale")
    parser.add_argument("--source", type=str, choices=["midas", "reco"], default="midas", help="Sorgente mappe: midas (default) o reco")
    parser.add_argument("--topo-veto-file", type=str, default=None, help="File .npz della maschera di veto topologico statico")
    parser.add_argument("--edge-margin", type=float, default=100.0, help="Margine esterno in pixel da escludere su tutti e 4 i lati (default: 100.0, 0 per disabilitare)")

    parser.add_argument("--no-dynamic-spark-veto", action="store_true", help="Disabilita la rimozione dinamica delle spike per singolo run")
    parser.add_argument("--spark-nsigma", type=float, default=20.0, help="Soglia core in sigma robuste per spike per-run su prof2d (default: 20.0)")
    parser.add_argument("--spark-skirt-nsigma", type=float, default=12.0, help="Soglia minima di coda in sigma robuste attorno al core (default: 12.0)")
    parser.add_argument("--spark-max-radius", type=int, default=4, help="Raggio massimo di crescita in bin dal nucleo della spike (default: 4)")
    parser.add_argument("--spark-hard-cap", type=float, default=150.0, help="Soglia alta fissa di sicurezza in ADC/pixel oltre cui ogni bin viene sempre tagliato (default: 150.0, <=0 per disabilitare)")
    parser.add_argument("--spark-ratio", type=float, default=None, help="Soglia alternativa come multiplo della mediana del singolo prof2d")
    parser.add_argument("--spark-abs", type=float, default=None, help="Soglia assoluta alternativa in ADC/pixel sul singolo prof2d")
    parser.add_argument("--spark-dilate", type=int, default=1, help="Dilatazione finale in bin attorno alle spike per-run (default: 1)")
    parser.add_argument("--spark-check-dir", type=str, default=None, help="Cartella dove salvare i plot diagnostici dei run con spike rimosse")

    parser.add_argument("--run-description", type=str, nargs="*", default=None, help="Valore/i di run_description da selezionare (default: nessun filtro)")
    parser.add_argument("--include-pedestals", action="store_true", help="Include anche i run di piedistallo se --run-description non è specificato")
    parser.add_argument("--group-by-desc", action="store_true", help="Separa le curve del grafico per run_description")
    parser.add_argument("--run-min", type=int, default=None, help="Numero del run minimo da includere")
    parser.add_argument("--run-max", type=int, default=None, help="Numero del run massimo da includere")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Valore di file_s3_tag da selezionare (default: LNGS)")
    parser.add_argument("--mode", type=str, choices=["per-run", "consecutive"], default="per-run", help="Modalità di analisi: per-run o consecutive")
    parser.add_argument("--output-dir", type=str, default=".", help="Cartella contenente i file ROOT delle mappe")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv", help="CSV di output con medie e tempi")
    parser.add_argument("--plot-output", type=str, default="trend_integral.png", help="File immagine di output per il plot")
    parser.add_argument("--time-col", type=str, default=None, help="Nome della colonna start-time nel CSV (opzionale, auto-rilevata)")
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours", help="Unità asse X (default: hours)")

    args = parser.parse_args()

    run_desc = args.run_description
    if run_desc is not None and len(run_desc) == 1:
        if "," in run_desc[0]:
            run_desc = [d.strip() for d in run_desc[0].split(",")]
        else:
            run_desc = run_desc[0]

    analyze_and_plot(
        csv_file=args.csv,
        source=args.source,
        topo_veto_file=args.topo_veto_file,
        edge_margin=args.edge_margin,
        dynamic_spark_veto=not args.no_dynamic_spark_veto,
        spark_nsigma=args.spark_nsigma,
        spark_skirt_nsigma=args.spark_skirt_nsigma,
        spark_max_radius=args.spark_max_radius,
        spark_hard_cap=args.spark_hard_cap,
        spark_ratio=args.spark_ratio,
        spark_abs=args.spark_abs,
        spark_dilate=args.spark_dilate,
        spark_check_dir=args.spark_check_dir,
        run_description=run_desc,
        exclude_pedestals=not args.include_pedestals,
        group_by_desc=args.group_by_desc,
        run_min=args.run_min,
        run_max=args.run_max,
        file_s3_tag=args.file_s3_tag,
        mode=args.mode,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        plot_output=args.plot_output,
        time_col=args.time_col,
        time_unit=args.time_unit,
    )