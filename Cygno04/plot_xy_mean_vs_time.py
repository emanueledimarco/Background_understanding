import os
import argparse
import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_map_filename(run_min, run_max, vgem, vdrift):
    return f"cygno04_sideA_muons_withVignetting_VGEM_{vgem}_VD_{vdrift}_run_{run_min}_{run_max}.root"


def detect_time_column(df, user_col=None):
    """
    Individua automaticamente la colonna dello start-time nel CSV iniziale.
    """
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
    """
    Converte la colonna temporale in datetime UTC, gestendo sia stringhe data/ora
    sia timestamp numerici Unix (in secondi o millisecondi).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum() and numeric.notna().any():
        median_val = numeric.dropna().median()
        unit = "ms" if median_val > 1e11 else "s"
        return pd.to_datetime(numeric, unit=unit, utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def find_consecutive_run_ranges_with_time(df):
    """
    Raggruppa run consecutivi con gli stessi valori di GEM1_V, DRIFT_V e run_description,
    conservando gli intervalli temporali dello start-time.
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
    current_t_start = df.loc[0, "_parsed_time"]
    current_t_end = df.loc[0, "_parsed_time"]

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]
        desc = row["run_description"] if "run_description" in df.columns else ""
        t_run = row["_parsed_time"]

        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift) and (desc == current_desc):
            current_end = run_num
            if pd.notnull(t_run):
                current_t_end = t_run
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
                "run_description": str(current_desc),
                "t_start": current_t_start,
                "t_end": current_t_end,
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift
            current_desc = desc
            current_t_start = t_run
            current_t_end = t_run

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
        "run_description": str(current_desc),
        "t_start": current_t_start,
        "t_end": current_t_end,
    })

    return ranges


def get_per_run_ranges_with_time(df):
    """
    Restituisce ogni singolo run come range autonomo [run, run].
    Permette di monitorare l'evoluzione temporale run per run.
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


def compute_xy_mean_from_root(root_path):
    """
    Legge prof2d e occ2d dal file ROOT e calcola la media complessiva in (x, y)
    sui bin accesi (occ2d > 0).
    """
    with uproot.open(root_path) as f:
        prof = f["prof2d"].values()
        occ = f["occ2d"].values()

    valid = occ > 0
    n_active = int(np.sum(valid))

    if n_active == 0:
        return {
            "mean_xy": np.nan,
            "std_xy": np.nan,
            "sem_xy": np.nan,
            "integral_xy": 0.0,
            "active_bins": 0,
        }

    prof_valid = prof[valid]
    mean_xy = float(np.mean(prof_valid))
    std_xy = float(np.std(prof_valid, ddof=1)) if n_active > 1 else 0.0
    sem_xy = float(std_xy / np.sqrt(n_active))
    integral_xy = float(np.sum(prof_valid))

    return {
        "mean_xy": mean_xy,
        "std_xy": std_xy,
        "sem_xy": sem_xy,
        "integral_xy": integral_xy,
        "active_bins": n_active,
    }


def analyze_and_plot(
    csv_file,
    run_description=None,
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
    print(f"ANALISI TEMPORALE E PLOT MEDIA IN (X, Y)")
    print(f"========================================================")

    df = pd.read_csv(csv_file)
    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()
    df["run_number"] = pd.to_numeric(df["run_number"], errors="coerce")
    df["GEM1_V"] = pd.to_numeric(df["GEM1_V"], errors="coerce")
    df["DRIFT_V"] = pd.to_numeric(df["DRIFT_V"], errors="coerce")

    # Filtro file_s3_tag se specificato
    mask = pd.Series(True, index=df.index)
    if file_s3_tag:
        target_tag = file_s3_tag.strip()
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
        run_ranges = find_consecutive_run_ranges_with_time(df_filtered)
    else:
        run_ranges = get_per_run_ranges_with_time(df_filtered)

    rows = []
    for r in run_ranges:
        r_min = r["run_min"]
        r_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]
        desc = r.get("run_description", "")

        fname = get_map_filename(r_min, r_max, vgem, vdrift)
        fpath = os.path.join(output_dir, fname)

        if not (os.path.exists(fpath) and os.path.getsize(fpath) > 0):
            print(f"[WARNING] File ROOT mancante o vuoto, saltato: {fpath}")
            continue

        try:
            stats = compute_xy_mean_from_root(fpath)
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
    cols_print = ["run_min", "run_max", "run_description", "VGEM", "DRIFT_V", f"dt_mid_{time_unit}", "mean_xy", "sem_xy", "integral_xy"]
    cols_to_show = [c for c in cols_print if c in out_df.columns]
    print(out_df[cols_to_show].to_string(index=False))
    print(f"\n--> Tabella salvata in: {summary_csv}")

    # Plot con pannello singolo
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Palette colori e marcatori per distinguere le categorie
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "<"]

    grouped = out_df.groupby(["run_description", "VGEM", "DRIFT_V"], sort=False)
    for idx, ((desc, vgem, vdrift), grp) in enumerate(grouped):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        x = grp[f"dt_mid_{time_unit}"].to_numpy()
        xerr_low = x - grp[f"dt_start_{time_unit}"].to_numpy()
        xerr_high = grp[f"dt_end_{time_unit}"].to_numpy() - x
        y_mean = grp["mean_xy"].to_numpy()
        y_err = grp["sem_xy"].to_numpy()

        label = f"{desc} (VGEM={vgem}V, VD={vdrift}V)"

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

    ax.set_title("Light integral history", fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Light integral (ADC/pixel)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(frameon=True, fontsize=10, loc="best")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_output)), exist_ok=True)
    plt.savefig(plot_output)
    plt.close(fig)
    print(f"--> Plot salvato in: {plot_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola la media e l'integrale delle mappe CYGNO e genera il plot vs tempo.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV iniziale")
    parser.add_argument("--run-description", type=str, nargs="*", default=None, help="Valore/i di run_description da selezionare (default: tutti)")
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

    # Normalizza run_description se passato come lista con virgole o singolo elemento
    run_desc = args.run_description
    if run_desc is not None and len(run_desc) == 1:
        if "," in run_desc[0]:
            run_desc = [d.strip() for d in run_desc[0].split(",")]
        else:
            run_desc = run_desc[0]

    analyze_and_plot(
        csv_file=args.csv,
        run_description=run_desc,
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
