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
    Raggruppa run consecutivi con gli stessi valori di GEM1_V e DRIFT_V,
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
    current_t_start = df.loc[0, "_parsed_time"]
    current_t_end = df.loc[0, "_parsed_time"]

    for i in range(1, len(df)):
        row = df.loc[i]
        run_num = row["run_number"]
        vgem = row["GEM1_V"]
        vdrift = row["DRIFT_V"]
        t_run = row["_parsed_time"]

        if (run_num == current_end + 1) and (vgem == current_vgem) and (vdrift == current_vdrift):
            current_end = run_num
            if pd.notnull(t_run):
                current_t_end = t_run
        else:
            ranges.append({
                "run_min": int(current_start),
                "run_max": int(current_end),
                "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
                "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
                "t_start": current_t_start,
                "t_end": current_t_end,
            })
            current_start = run_num
            current_end = run_num
            current_vgem = vgem
            current_vdrift = vdrift
            current_t_start = t_run
            current_t_end = t_run

    ranges.append({
        "run_min": int(current_start),
        "run_max": int(current_end),
        "vgem": int(current_vgem) if pd.notnull(current_vgem) else current_vgem,
        "vdrift": int(current_vdrift) if pd.notnull(current_vdrift) else current_vdrift,
        "t_start": current_t_start,
        "t_end": current_t_end,
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
    run_description="MuonMap",
    file_s3_tag="LNGS",
    output_dir=".",
    summary_csv="xy_mean_summary.csv",
    plot_output="xy_mean_vs_time.png",
    time_col=None,
    time_unit="hours",
):
    print(f"\n========================================================")
    print(f"ANALISI TEMPORALE E PLOT MEDIA IN (X, Y)")
    print(f"========================================================")

    df = pd.read_csv(csv_file)
    df["run_description"] = df["run_description"].astype(str).str.strip()
    df["file_s3_tag"] = df["file_s3_tag"].astype(str).str.strip()

    target_desc = run_description.strip()
    target_tag = file_s3_tag.strip()

    mask = (df["run_description"] == target_desc) & (df["file_s3_tag"] == target_tag)
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        print(f"Nessun run trovato nel CSV con run_description='{target_desc}' e file_s3_tag='{target_tag}'.")
        return

    df_filtered["run_number"] = pd.to_numeric(df_filtered["run_number"], errors="coerce")
    df_filtered["GEM1_V"] = pd.to_numeric(df_filtered["GEM1_V"], errors="coerce")
    df_filtered["DRIFT_V"] = pd.to_numeric(df_filtered["DRIFT_V"], errors="coerce")

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

    run_ranges = find_consecutive_run_ranges_with_time(df_filtered)

    rows = []
    for r in run_ranges:
        run_min = r["run_min"]
        run_max = r["run_max"]
        vgem = r["vgem"]
        vdrift = r["vdrift"]

        fname = get_map_filename(run_min, run_max, vgem, vdrift)
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
            "run_min": run_min,
            "run_max": run_max,
            "VGEM": vgem,
            "DRIFT_V": vdrift,
            "t_start_utc": r["t_start"].isoformat(),
            "t_end_utc": r["t_end"].isoformat(),
            f"dt_start_{time_unit}": dt_start,
            f"dt_mid_{time_unit}": dt_mid,
            f"dt_end_{time_unit}": dt_end,
            "file": fname,
            **stats,
        })

    if not rows:
        print("Nessun dato valido disponibile per il summary o il plot.")
        return

    out_df = pd.DataFrame(rows).sort_values(f"dt_start_{time_unit}").reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_csv)), exist_ok=True)
    out_df.to_csv(summary_csv, index=False)

    print("\nRiepilogo calcolato:")
    cols_print = ["run_min", "run_max", "VGEM", "DRIFT_V", f"dt_mid_{time_unit}", "mean_xy", "sem_xy"]
    print(out_df[cols_print].to_string(index=False))
    print(f"\n--> Tabella salvata in: {summary_csv}")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

    grouped = out_df.groupby(["VGEM", "DRIFT_V"], sort=False)
    for (vgem, vdrift), grp in grouped:
        x = grp[f"dt_mid_{time_unit}"].to_numpy()
        xerr_low = x - grp[f"dt_start_{time_unit}"].to_numpy()
        xerr_high = grp[f"dt_end_{time_unit}"].to_numpy() - x
        y = grp["mean_xy"].to_numpy()
        yerr = grp["sem_xy"].to_numpy()

        label = f"VGEM = {vgem} V, VD = {vdrift} V"
        ax.errorbar(
            x,
            y,
            xerr=[xerr_low, xerr_high],
            yerr=yerr,
            fmt="o-",
            capsize=3,
            markersize=6,
            linewidth=1.3,
            label=label,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Media spaziale prof2d [ADC / pixel]", fontsize=12)
    ax.set_title(f"Evoluzione temporale della risposta media in (x, y) - {target_desc}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_output)), exist_ok=True)
    plt.savefig(plot_output)
    plt.close(fig)
    print(f"--> Plot salvato in: {plot_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola la media in (x, y) delle mappe CYGNO e genera il plot vs tempo.")
    parser.add_argument("--csv", type=str, required=True, help="Path del file CSV iniziale")
    parser.add_argument("--run-description", type=str, default="MuonMap", help="Valore di run_description da selezionare")
    parser.add_argument("--file-s3-tag", type=str, default="LNGS", help="Valore di file_s3_tag da selezionare")
    parser.add_argument("--output-dir", type=str, default=".", help="Cartella contenente i file ROOT delle mappe")
    parser.add_argument("--summary-csv", type=str, default="xy_mean_summary.csv", help="CSV di output con medie e tempi")
    parser.add_argument("--plot-output", type=str, default="xy_mean_vs_time.png", help="File immagine di output per il plot")
    parser.add_argument("--time-col", type=str, default=None, help="Nome della colonna start-time nel CSV (opzionale, auto-rilevata)")
    parser.add_argument("--time-unit", type=str, choices=["seconds", "hours", "days"], default="hours", help="Unità asse X (default: hours)")

    args = parser.parse_args()

    analyze_and_plot(
        csv_file=args.csv,
        run_description=args.run_description,
        file_s3_tag=args.file_s3_tag,
        output_dir=args.output_dir,
        summary_csv=args.summary_csv,
        plot_output=args.plot_output,
        time_col=args.time_col,
        time_unit=args.time_unit,
    )