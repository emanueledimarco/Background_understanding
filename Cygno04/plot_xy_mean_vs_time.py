#!/usr/bin/env python3
import os
import glob
import argparse
from collections import deque
import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PED_REGEX = r"\bped(?:estal|s)?\b"


def save_fig_png_pdf(fig, filepath):
    base, ext = os.path.splitext(filepath)
    if ext.lower() not in (".png", ".pdf"):
        base = filepath
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    png_path = f"{base}.png"
    pdf_path = f"{base}.pdf"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
    return png_path, pdf_path


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

    return binary_dilate_2d(grown_union, radius=dilate_bins) & base_valid


def parse_z_max_arg(z_max_str):
    if z_max_str is None:
        return "auto"
    val = str(z_max_str).strip().lower()
    if val in ("auto", "mean"):
        return "auto"
    if val in ("free", "none", "off"):
        return None
    num = float(val)
    return num if num > 0 else None


def detect_time_column(df, user_time_col=None):
    if user_time_col is not None:
        if user_time_col not in df.columns:
            raise ValueError(f"Colonna tempo '{user_time_col}' non trovata nel CSV. Colonne: {list(df.columns)}")
        return user_time_col

    candidates = [
        "start_time", "start-time", "starttime", "start_date", "startdate",
        "timestamp", "time_start", "begin_time", "time", "date", "datetime",
    ]
    lower_map = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for col in df.columns:
        if "start" in col.lower() and ("time" in col.lower() or "date" in col.lower()):
            return col
    return None


def parse_timestamps(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series) and len(series) > 0:
        med = float(numeric.median())
        if med > 1e14:
            return pd.to_datetime(numeric, unit="ns", utc=True)
        if med > 1e11:
            return pd.to_datetime(numeric, unit="ms", utc=True)
        return pd.to_datetime(numeric, unit="s", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_run_blocks(df_filtered, mode="per-run"):
    df_s = df_filtered.sort_values("run_number").reset_index(drop=True)
    blocks = []

    if mode == "per-run":
        for _, row in df_s.iterrows():
            rnum = int(row["run_number"])
            blocks.append({
                "run_min": rnum,
                "run_max": rnum,
                "vgem": row.get("GEM1_V", np.nan),
                "vdrift": row.get("DRIFT_V", np.nan),
                "run_description": str(row.get("run_description", "")).strip(),
                "t_start": row["_parsed_time"],
                "t_end": row["_parsed_time"],
            })
        return blocks

    current = None
    for _, row in df_s.iterrows():
        rnum = int(row["run_number"])
        vgem = row.get("GEM1_V", np.nan)
        vdrift = row.get("DRIFT_V", np.nan)
        desc = str(row.get("run_description", "")).strip()
        ts = row["_parsed_time"]

        same_cfg = (
            current is not None
            and rnum == current["run_max"] + 1
            and np.isclose(vgem, current["vgem"], equal_nan=True)
            and np.isclose(vdrift, current["vdrift"], equal_nan=True)
            and desc == current["run_description"]
        )
        if same_cfg:
            current["run_max"] = rnum
            if pd.notna(ts):
                current["t_end"] = ts
        else:
            if current is not None:
                blocks.append(current)
            current = {
                "run_min": rnum,
                "run_max": rnum,
                "vgem": vgem,
                "vdrift": vdrift,
                "run_description": desc,
                "t_start": ts,
                "t_end": ts,
            }
    if current is not None:
        blocks.append(current)
    return blocks


def find_root_file_for_block(output_dir, source, block, use_topoveto=False):
    r0, r1 = block["run_min"], block["run_max"]
    vgem, vdrift = block["vgem"], block["vdrift"]

    veto_tag = "_topoveto" if use_topoveto else ""
    vgem_str = f"{int(vgem)}" if pd.notna(vgem) and float(vgem).is_integer() else f"{vgem}"
    vd_str = f"{int(vdrift)}" if pd.notna(vdrift) and float(vdrift).is_integer() else f"{vdrift}"

    if source == "midas":
        exact = os.path.join(output_dir, f"cygno04_sideA_midas_raw{veto_tag}_VGEM_{vgem_str}_VD_{vd_str}_run_{r0}_{r1}.root")
    else:
        exact = os.path.join(output_dir, f"cygno04_sideA_muons_withVignetting{veto_tag}_VGEM_{vgem_str}_VD_{vd_str}_run_{r0}_{r1}.root")

    if os.path.exists(exact) and os.path.getsize(exact) > 0:
        return exact

    matches = sorted(glob.glob(os.path.join(output_dir, f"*_run_{r0}_{r1}.root")))
    matches = [
        m for m in matches
        if "_bare_" not in os.path.basename(m)
        and not m.endswith("_clean.root")
        and os.path.getsize(m) > 0
    ]
    if matches:
        return matches[0]
    return None


def load_and_clean_map(
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
):
    with uproot.open(root_path) as f:
        if source == "midas":
            prof_key = "prof2d_images" if "prof2d_images" in f else "prof2d"
            int_key = "int2d_images" if "int2d_images" in f else "int2d"
            occ_key = "occ2d_images" if "occ2d_images" in f else "occ2d"
        else:
            prof_key, int_key, occ_key = "prof2d", "int2d", "occ2d"

        prof, x_edges, y_edges = f[prof_key].to_numpy()
        integ = f[int_key].values()
        occ = f[occ_key].values()

    base_valid = (occ > 0) & np.isfinite(prof)
    if topo_veto_mask is not None:
        if topo_veto_mask.shape != prof.shape:
            raise ValueError(f"Shape topo_veto_mask {topo_veto_mask.shape} != prof {prof.shape}")
        base_valid &= (~topo_veto_mask)

    if edge_margin is not None and edge_margin > 0:
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        x_in = (x_centers >= (x_edges[0] + edge_margin)) & (x_centers <= (x_edges[-1] - edge_margin))
        y_in = (y_centers >= (y_edges[0] + edge_margin)) & (y_centers <= (y_edges[-1] - edge_margin))
        base_valid &= np.outer(x_in, y_in)

    max_before_clean = float(np.max(prof[base_valid])) if np.any(base_valid) else np.nan
    dyn_veto_mask = np.zeros_like(base_valid, dtype=bool)
    core_thr = np.nan

    if dynamic_spark_veto and np.any(base_valid):
        vals = prof[base_valid]
        median_val = float(np.median(vals))
        mad_val = float(np.median(np.abs(vals - median_val)))
        robust_std = 1.4826 * mad_val
        if robust_std <= 0:
            q75, q25 = np.percentile(vals, [75, 25])
            robust_std = float((q75 - q25) / 1.349) if q75 > q25 else float(np.std(vals))

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

        dyn_veto_mask = detect_bounded_spark_mask(
            prof=prof,
            base_valid=base_valid,
            core_thr=core_thr,
            base_skirt_thr=base_skirt_thr,
            peak_fraction=0.02,
            max_grow_radius=spark_max_radius,
            dilate_bins=spark_dilate,
        )

        if has_hard_cap:
            hard_hits = base_valid & (prof > float(spark_hard_cap))
            if np.any(hard_hits):
                dyn_veto_mask |= binary_dilate_2d(hard_hits, radius=spark_dilate) & base_valid

    clean_valid = base_valid & (~dyn_veto_mask)
    n_active = int(np.sum(clean_valid))

    if n_active > 0:
        prof_v = prof[clean_valid]
        int_v = integ[clean_valid]
        occ_v = occ[clean_valid]

        mean_xy = float(np.mean(prof_v))
        std_xy = float(np.std(prof_v, ddof=1)) if n_active > 1 else 0.0
        sem_xy = float(std_xy / np.sqrt(n_active))
        tot_occ = float(np.sum(occ_v))
        weighted_mean_xy = float(np.sum(int_v) / tot_occ) if tot_occ > 0 else np.nan
        integral_xy = float(np.sum(prof_v))
        max_after_clean = float(np.max(prof_v))
    else:
        mean_xy = np.nan
        sem_xy = np.nan
        weighted_mean_xy = np.nan
        integral_xy = 0.0
        max_after_clean = np.nan

    return {
        "prof": prof,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "base_valid": base_valid,
        "dyn_veto_mask": dyn_veto_mask,
        "clean_valid": clean_valid,
        "core_thr": core_thr,
        "mean_xy": mean_xy,
        "sem_xy": sem_xy,
        "weighted_mean_xy": weighted_mean_xy,
        "integral_xy": integral_xy,
        "active_bins": n_active,
        "dynamic_veto_bins": int(np.sum(dyn_veto_mask)),
        "max_before_clean": max_before_clean,
        "max_after_clean": max_after_clean,
    }


def save_2d_map_plot(
    map_data,
    title,
    out_base,
    z_min=0.0,
    z_max=None,
):
    prof_clean = np.where(map_data["clean_valid"], map_data["prof"], np.nan).T
    x_edges = map_data["x_edges"]
    y_edges = map_data["y_edges"]

    vmax_eff = z_max if (z_max is not None and np.isfinite(z_max) and z_max > z_min) else None

    fig, ax = plt.subplots(figsize=(7.2, 9.2), dpi=150)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        prof_clean,
        shading="flat",
        cmap="rainbow",
        vmin=z_min,
        vmax=vmax_eff,
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Mean Charge / Hit [ADC]", fontsize=10)

    max_val = map_data["max_after_clean"]
    mean_val = map_data["mean_xy"]
    zmax_label = f"{vmax_eff:.1f}" if vmax_eff is not None else "auto"
    ax.set_title(
        f"{title}\nmean={mean_val:.2f} ADC | max={max_val:.1f} ADC | zmax={zmax_label}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X [pixel]", fontsize=10)
    ax.set_ylabel("Y [pixel]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3, color="gray")

    plt.tight_layout()
    save_fig_png_pdf(fig, out_base)
    plt.close(fig)


def save_spark_check_plot(
    map_data,
    run_label,
    check_dir,
    z_min=0.0,
    z_max=None,
):
    out_base = os.path.join(check_dir, f"spark_check_{run_label}")

    x_edges = map_data["x_edges"]
    y_edges = map_data["y_edges"]
    prof_before = np.where(map_data["base_valid"], map_data["prof"], np.nan).T
    prof_after = np.where(map_data["clean_valid"], map_data["prof"], np.nan).T
    veto_overlay = np.where(map_data["dyn_veto_mask"], 1.0, np.nan).T

    vmax_clean = z_max if (z_max is not None and np.isfinite(z_max) and z_max > z_min) else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), dpi=150)

    m0 = axes[0].pcolormesh(x_edges, y_edges, prof_before, shading="flat", cmap="rainbow")
    cb0 = fig.colorbar(m0, ax=axes[0], pad=0.02, fraction=0.046)
    cb0.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[0].set_title(f"1. Prima pulizia prof2d (max={map_data['max_before_clean']:.0f} ADC)", fontsize=11, fontweight="bold")

    axes[1].pcolormesh(x_edges, y_edges, prof_after, shading="flat", cmap="Greys", alpha=0.55)
    axes[1].pcolormesh(x_edges, y_edges, veto_overlay, shading="flat", cmap="autumn", vmin=0, vmax=1, alpha=0.95)
    axes[1].set_title(
        f"2. Spike rimosse ({map_data['dynamic_veto_bins']} bin | thr={map_data['core_thr']:.1f})",
        fontsize=11,
        fontweight="bold",
    )

    m2 = axes[2].pcolormesh(
        x_edges,
        y_edges,
        prof_after,
        shading="flat",
        cmap="rainbow",
        vmin=z_min,
        vmax=vmax_clean,
    )
    cb2 = fig.colorbar(m2, ax=axes[2], pad=0.02, fraction=0.046)
    cb2.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[2].set_title(f"3. Mappa pulita (max={map_data['max_after_clean']:.1f} ADC)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("X [pixel]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    axes[0].set_ylabel("Y [pixel]", fontsize=10)

    fig.suptitle(f"Pulizia dinamica prof2d — {run_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig_png_pdf(fig, out_base)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Calcola la history temporale dell'integrale medio in (x, y) e genera le mappe 2D.")
    parser.add_argument("--csv", type=str, required=True, help="CSV dei run")
    parser.add_argument("--source", type=str, choices=["midas", "reco"], default="midas")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS")
    parser.add_argument("--run-min", type=int, default=None)
    parser.add_argument("--run-max", type=int, default=None)
    parser.add_argument("--run-description", type=str, nargs="*", default=None)
    parser.add_argument("--include-pedestals", action="store_true")
    parser.add_argument("--group-by-desc", action="store_true")
    parser.add_argument("--mode", type=str, choices=["per-run", "consecutive"], default="per-run")

    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv")
    parser.add_argument("--plot-output", type=str, default="trend_integral.png")
    parser.add_argument("--plot-2dmaps", action="store_true", help="Salva i plot 2D PNG e PDF per ciascun run")
    parser.add_argument("--plots-2d-dir", type=str, default="plots_2d", help="Cartella di output per le mappe 2D")
    parser.add_argument("--z-min", type=float, default=0.0, help="Limite inferiore scala colori Z delle mappe 2D (default: 0.0)")
    parser.add_argument(
        "--z-max",
        type=str,
        default="auto",
        help="Limite superiore scala colori Z delle mappe 2D: 'auto' (media dei max_after_clean, default), valore numerico (es. 55.0), o 'free'",
    )

    parser.add_argument("--topo-veto-file", type=str, default=None)
    parser.add_argument("--edge-margin", type=float, default=100.0)
    parser.add_argument("--no-dynamic-spark-veto", action="store_true")
    parser.add_argument("--spark-nsigma", type=float, default=20.0)
    parser.add_argument("--spark-skirt-nsigma", type=float, default=12.0)
    parser.add_argument("--spark-max-radius", type=int, default=4)
    parser.add_argument("--spark-hard-cap", type=float, default=150.0)
    parser.add_argument("--spark-ratio", type=float, default=None)
    parser.add_argument("--spark-abs", type=float, default=None)
    parser.add_argument("--spark-dilate", type=int, default=1)
    parser.add_argument("--spark-check-dir", type=str, default=None)

    parser.add_argument("--time-col", type=str, default=None)
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours")

    args = parser.parse_args()
    z_max_cfg = parse_z_max_arg(args.z_max)

    topo_veto_mask = None
    if args.topo_veto_file is not None:
        if not os.path.exists(args.topo_veto_file):
            raise FileNotFoundError(f"File veto topologico non trovato: {args.topo_veto_file}")
        topo_veto_mask = np.load(args.topo_veto_file)["veto_mask"].astype(bool)

    df = pd.read_csv(args.csv)
    df["run_number"] = pd.to_numeric(df["run_number"], errors="coerce")
    df = df.dropna(subset=["run_number"]).copy()
    df["run_number"] = df["run_number"].astype(int)

    if "file_s3_tag" in df.columns and args.file_s3_tag:
        df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()
        df = df[df["file_s3_tag"] == args.file_s3_tag].copy()

    if "run_description" in df.columns:
        df["run_description"] = df["run_description"].astype(str).str.strip()
        if args.run_description:
            desc_list = []
            for item in args.run_description:
                desc_list.extend([x.strip() for x in item.split(",") if x.strip()])
            df = df[df["run_description"].isin(desc_list)].copy()
        elif not args.include_pedestals:
            is_ped = df["run_description"].str.lower().str.contains(PED_REGEX, regex=True, na=False)
            if "pedestal_run" in df.columns:
                ped_col = pd.to_numeric(df["pedestal_run"], errors="coerce")
                is_ped |= (df["run_number"] == ped_col)
            df = df[~is_ped].copy()

    if args.run_min is not None:
        df = df[df["run_number"] >= args.run_min].copy()
    if args.run_max is not None:
        df = df[df["run_number"] <= args.run_max].copy()

    if df.empty:
        print("Nessun run corrispondente ai filtri nel CSV.")
        return

    for col in ("GEM1_V", "DRIFT_V"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    time_col = detect_time_column(df, args.time_col)
    if time_col is not None:
        df["_parsed_time"] = parse_timestamps(df[time_col])
    else:
        df["_parsed_time"] = pd.NaT

    blocks = build_run_blocks(df, mode=args.mode)
    valid_times = [b["t_start"] for b in blocks if pd.notna(b["t_start"])]
    t0 = min(valid_times) if valid_times else None

    scale_div = {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[args.time_unit]

    rows = []
    cached_maps = []

    for blk in blocks:
        r0, r1 = blk["run_min"], blk["run_max"]
        root_path = find_root_file_for_block(
            args.output_dir,
            args.source,
            blk,
            use_topoveto=(args.topo_veto_file is not None),
        )
        if root_path is None:
            print(f"[SKIP] File ROOT non trovato per run {r0}..{r1}")
            continue

        map_data = load_and_clean_map(
            root_path=root_path,
            source=args.source,
            topo_veto_mask=topo_veto_mask,
            edge_margin=args.edge_margin,
            dynamic_spark_veto=not args.no_dynamic_spark_veto,
            spark_nsigma=args.spark_nsigma,
            spark_skirt_nsigma=args.spark_skirt_nsigma,
            spark_max_radius=args.spark_max_radius,
            spark_hard_cap=args.spark_hard_cap,
            spark_ratio=args.spark_ratio,
            spark_abs=args.spark_abs,
            spark_dilate=args.spark_dilate,
        )

        if t0 is not None and pd.notna(blk["t_start"]):
            dt_start = (blk["t_start"] - t0).total_seconds() / scale_div
            dt_end = (blk["t_end"] - t0).total_seconds() / scale_div
            dt_mid = 0.5 * (dt_start + dt_end)
        else:
            dt_start = float(r0)
            dt_end = float(r1)
            dt_mid = 0.5 * (dt_start + dt_end)

        row_dict = {
            "run_min": r0,
            "run_max": r1,
            "VGEM": blk["vgem"],
            "DRIFT_V": blk["vdrift"],
            "run_description": blk["run_description"],
            "t_start": dt_start,
            "t_end": dt_end,
            "t_mid": dt_mid,
            "mean_xy": map_data["mean_xy"],
            "sem_xy": map_data["sem_xy"],
            "weighted_mean_xy": map_data["weighted_mean_xy"],
            "integral_xy": map_data["integral_xy"],
            "active_bins": map_data["active_bins"],
            "dynamic_veto_bins": map_data["dynamic_veto_bins"],
            "max_before_clean": map_data["max_before_clean"],
            "max_after_clean": map_data["max_after_clean"],
            "file": os.path.basename(root_path),
        }
        rows.append(row_dict)
        cached_maps.append((row_dict, map_data))

    if not rows:
        print("Nessun file ROOT trovato per generare summary e plot.")
        return

    out_df = pd.DataFrame(rows)
    summary_path = args.summary_csv if os.path.isabs(args.summary_csv) else os.path.join(args.output_dir, args.summary_csv)
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    out_df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] Salvato CSV riepilogo in: {summary_path}")

    # Risoluzione z_max uniforme su tutte le mappe 2D
    if z_max_cfg == "auto":
        valid_maxes = out_df["max_after_clean"].dropna().values
        resolved_zmax = float(np.mean(valid_maxes)) if len(valid_maxes) > 0 else None
        if resolved_zmax is not None:
            print(
                f"[2D MAPS] --z-max auto -> Z range uniforme: [{args.z_min:.1f}, {resolved_zmax:.1f}] ADC/pixel "
                f"(media dei max_after_clean su {len(valid_maxes)} mappe)"
            )
    else:
        resolved_zmax = z_max_cfg
        if resolved_zmax is not None:
            print(f"[2D MAPS] --z-max fisso -> Z range uniforme: [{args.z_min:.1f}, {resolved_zmax:.1f}] ADC/pixel")
        else:
            print("[2D MAPS] --z-max free -> Z range libero per ciascuna mappa")

    # Generazione plot 2D e spark check (PNG + PDF)
    if args.plot_2dmaps or args.spark_check_dir is not None:
        for row_dict, map_data in cached_maps:
            r0, r1 = row_dict["run_min"], row_dict["run_max"]
            run_tag = f"run_{r0}" if r0 == r1 else f"run_{r0}_{r1}"

            if args.plot_2dmaps:
                out_2d_dir = args.plots_2d_dir if os.path.isabs(args.plots_2d_dir) else os.path.join(args.output_dir, args.plots_2d_dir)
                out_2d_base = os.path.join(out_2d_dir, f"map2d_{run_tag}")
                save_2d_map_plot(
                    map_data=map_data,
                    title=f"CYGNO 2D Light Map — {run_tag}",
                    out_base=out_2d_base,
                    z_min=args.z_min,
                    z_max=resolved_zmax,
                )

            if args.spark_check_dir is not None and map_data["dynamic_veto_bins"] > 0:
                save_spark_check_plot(
                    map_data=map_data,
                    run_label=run_tag,
                    check_dir=args.spark_check_dir,
                    z_min=args.z_min,
                    z_max=resolved_zmax,
                )

        if args.plot_2dmaps:
            print(f"[2D MAPS] Salvate {len(cached_maps)} mappe 2D (PNG e PDF) in: {args.plots_2d_dir}")

    # Plot history vs tempo (punti neri, scala Y +12%, numeri di run in alto, senza titolo)
    plot_path = args.plot_output if os.path.isabs(args.plot_output) else os.path.join(args.output_dir, args.plot_output)

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=150)

    if args.group_by_desc and "run_description" in out_df.columns:
        groups = list(out_df.groupby("run_description", dropna=False))
    else:
        groups = list(out_df.groupby(["VGEM", "DRIFT_V"], dropna=False))

    palette = ["black", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "<"]

    for idx, (key, grp) in enumerate(groups):
        grp_s = grp.sort_values("t_mid")
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]

        if isinstance(key, tuple):
            vgem_v, vd_v = key
            label = f"VGEM={vgem_v:.0f} V, VD={vd_v:.0f} V" if pd.notna(vgem_v) and pd.notna(vd_v) else "All runs"
        else:
            label = str(key)

        x = grp_s["t_mid"].to_numpy()
        xerr_low = x - grp_s["t_start"].to_numpy()
        xerr_high = grp_s["t_end"].to_numpy() - x
        use_xerr = args.mode == "consecutive" and np.any((xerr_low > 0) | (xerr_high > 0))

        ax.errorbar(
            x,
            grp_s["mean_xy"].to_numpy(),
            xerr=[xerr_low, xerr_high] if use_xerr else None,
            yerr=grp_s["sem_xy"].to_numpy(),
            fmt=f"{marker}-",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5.5,
            linewidth=1.3,
            label=label,
        )

    xlabel = f"Time from first run [{args.time_unit}]" if t0 is not None else "Run number"
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Mean Charge / Pixel [ADC]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9.5, frameon=True)

    # Espansione scala Y (~12% in più)
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    if yrange > 0:
        ax.set_ylim(ymin - 0.04 * yrange, ymax + 0.08 * yrange)

    # Scala superiore con i numeri di run
    out_df_sorted = out_df.sort_values("t_mid").reset_index(drop=True)
    tick_positions = out_df_sorted["t_mid"].to_numpy()
    tick_labels = [
        f"{int(r0)}" if int(r0) == int(r1) else f"{int(r0)}-{int(r1)}"
        for r0, r1 in zip(out_df_sorted["run_min"], out_df_sorted["run_max"])
    ]
    if len(tick_positions) > 45:
        step = int(np.ceil(len(tick_positions) / 40.0))
        tick_labels = [lbl if (i % step == 0) else "" for i, lbl in enumerate(tick_labels)]

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(tick_positions)
    ax_top.set_xticklabels(tick_labels, fontsize=7.5, rotation=45, ha="left")
    ax_top.set_xlabel("Run number", fontsize=10, labelpad=8)

    plt.tight_layout()
    png_out, pdf_out = save_fig_png_pdf(fig, plot_path)
    plt.close(fig)
    print(f"[HISTORY PLOT] Salvato grafico temporale in: {png_out} e {pdf_out}")


if __name__ == "__main__":
    main()