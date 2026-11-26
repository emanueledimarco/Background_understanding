import argparse
import os
import pickle
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import uproot
import yaml

from data_reading.read_data_2D import selection_cfg

PKL_CLEAN = "data/cache/cygno_clusters_dataset_nopileup.pkl"
PKL_RAW = "data/cache/cygno_clusters_dataset.pkl"
DEFAULT_PKL = PKL_CLEAN if os.path.exists(PKL_CLEAN) else PKL_RAW

CFG_YAML = "flow_configuration.yaml"
PLOT_OUT = "check_bimodal_keys.png"


def _extract_run_id(filepath):
    m = re.search(r"run(\d+)", os.path.basename(filepath))
    return m.group(1) if m else os.path.basename(filepath)


def _smooth_1d(y, sigma_bins=1.5):
    r = int(max(3, round(3 * sigma_bins)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def _count_density_peaks(x):
    """Conta quanti massimi locali ben separati esistono nell'istogramma smoothed."""
    if len(x) < 40:
        return 1
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        return 1
    bins = np.linspace(lo - 0.02, hi + 0.02, 55)
    hist, _ = np.histogram(x, bins=bins)
    sm = _smooth_1d(hist.astype(np.float64), sigma_bins=1.6)

    peaks = []
    for i in range(1, len(sm) - 1):
        if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]:
            peaks.append(i)
    if len(peaks) <= 1:
        return len(peaks)

    peaks = sorted(peaks, key=lambda idx: sm[idx], reverse=True)
    p0 = peaks[0]
    h0 = sm[p0]
    valid_peaks = [p0]
    for p in peaks[1:]:
        if sm[p] < 0.16 * h0:
            continue
        i_min, i_max = min(p0, p), max(p0, p)
        if i_max - i_min < 3:
            continue
        valley = float(np.min(sm[i_min : i_max + 1]))
        if valley < 0.85 * sm[p]:
            valid_peaks.append(p)
    return len(valid_peaks)


def fit_1d_gmm2(x, n_iter=80):
    """Fit 1D a 2 Gaussiane via EM vs 1 Gaussiana singola."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    mu0 = float(np.mean(x))
    s0 = float(max(np.std(x), 1e-4))
    ll1 = float(
        np.sum(-0.5 * np.log(2.0 * np.pi * s0**2) - 0.5 * ((x - mu0) / s0) ** 2)
    )
    bic1 = 2.0 * np.log(max(n, 2)) - 2.0 * ll1

    # Multi-start leggero sui quantili per beccare sia split 50/50 sia code 80/20
    best = None
    for q_lo, q_hi in [(25.0, 75.0), (15.0, 65.0), (35.0, 85.0)]:
        mu1, mu2 = [float(v) for v in np.percentile(x, [q_lo, q_hi])]
        s1 = s2 = float(max(s0 * 0.55, 0.02))
        w1, w2 = 0.5, 0.5
        s_floor = 0.015

        for _ in range(n_iter):
            p1 = w1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2) / (np.sqrt(2.0 * np.pi) * s1)
            p2 = w2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2) / (np.sqrt(2.0 * np.pi) * s2)
            tot = p1 + p2 + 1e-300
            r1 = p1 / tot
            r2 = p2 / tot

            n1 = float(np.sum(r1))
            n2 = float(np.sum(r2))
            if n1 < 5.0 or n2 < 5.0:
                break

            w1 = n1 / n
            w2 = n2 / n
            mu1 = float(np.sum(r1 * x) / n1)
            mu2 = float(np.sum(r2 * x) / n2)
            s1 = float(max(np.sqrt(np.sum(r1 * (x - mu1) ** 2) / n1), s_floor))
            s2 = float(max(np.sqrt(np.sum(r2 * (x - mu2) ** 2) / n2), s_floor))

        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
            s1, s2 = s2, s1
            w1, w2 = w2, w1

        p1 = w1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2) / (np.sqrt(2.0 * np.pi) * s1)
        p2 = w2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2) / (np.sqrt(2.0 * np.pi) * s2)
        ll2 = float(np.sum(np.log(p1 + p2 + 1e-300)))
        if best is None or ll2 > best["ll2"]:
            best = {
                "mu1": mu1,
                "mu2": mu2,
                "s1": s1,
                "s2": s2,
                "w1": w1,
                "w2": w2,
                "ll2": ll2,
            }

    bic2 = 5.0 * np.log(max(n, 2)) - 2.0 * best["ll2"]
    dmu = best["mu2"] - best["mu1"]
    ashman_d = dmu / np.sqrt(0.5 * (best["s1"] ** 2 + best["s2"] ** 2))
    delta_bic = bic1 - bic2
    n_peaks = _count_density_peaks(x)

    best.update(
        {
            "std_tot": s0,
            "dmu": dmu,
            "gain_ratio": 10.0**dmu,
            "ashman_d": ashman_d,
            "delta_bic": delta_bic,
            "n_peaks": n_peaks,
        }
    )
    return best


def load_datamap_files():
    """Legge data_map dal YAML e restituisce key -> lista di file ROOT."""
    data_map_path = None
    if os.path.exists(CFG_YAML):
        with open(CFG_YAML, "r") as f:
            cfg = yaml.safe_load(f)
        data_map_path = cfg.get("data_inputs", {}).get("data_map")

    if not data_map_path or not os.path.exists(data_map_path):
        for fn in sorted(os.listdir(".")):
            if "data" in fn.lower() and "map" in fn.lower() and fn.endswith((".yaml", ".yml")):
                data_map_path = fn
                break

    if not data_map_path or not os.path.exists(data_map_path):
        return {}, None

    with open(data_map_path, "r") as f:
        raw_map = yaml.safe_load(f)

    files_by_key = defaultdict(list)
    for raw_k, flist in raw_map.items():
        mk = tuple(round(float(x), 4) for x in str(raw_k).split(","))
        files_by_key[mk].extend(flist)
    return files_by_key, data_map_path


def inspect_remote_roots_for_key(files):
    """
    Legge solo i branch scalari di selezione dai file ROOT remoti delle chiavi sospette
    per confrontare il picco run per run.
    """
    file_stats = []
    for url in files:
        run_id = _extract_run_id(url)
        try:
            with uproot.open(url) as rf:
                arr = rf["Events"].arrays(
                    ["sc_integral", "sc_xmean", "sc_ymean", "sc_nhits"],
                    library="np",
                )
            ints = []
            for i in range(len(arr["sc_integral"])):
                sc_i = arr["sc_integral"][i]
                sc_x = arr["sc_xmean"][i]
                sc_y = arr["sc_ymean"][i]
                sc_n = arr["sc_nhits"][i]
                m = (
                    (sc_i >= selection_cfg["integral_min"])
                    & (sc_i <= selection_cfg["integral_max"])
                    & (sc_x >= selection_cfg["x_min"])
                    & (sc_x <= selection_cfg["x_max"])
                    & (sc_y >= selection_cfg["y_min"])
                    & (sc_y <= selection_cfg["y_max"])
                    & (sc_n >= selection_cfg["n_hits"])
                )
                ints.extend(sc_i[m])
            vals = np.log10(np.asarray(ints, dtype=np.float64))
            file_stats.append(
                {
                    "run": run_id,
                    "url": url,
                    "n": len(vals),
                    "median": float(np.median(vals)) if len(vals) > 0 else np.nan,
                    "mean": float(np.mean(vals)) if len(vals) > 0 else np.nan,
                    "vals": vals,
                }
            )
        except Exception as e:
            file_stats.append(
                {
                    "run": run_id,
                    "url": url,
                    "n": -1,
                    "median": np.nan,
                    "mean": np.nan,
                    "vals": None,
                    "err": str(e),
                }
            )
    return file_stats


def main():
    parser = argparse.ArgumentParser(description="Controlla distribuzioni bimodali nei dati dal .pkl")
    parser.add_argument("--pkl", type=str, default=DEFAULT_PKL, help="Path del dataset .pkl")
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="Non aprire i file ROOT remoti per le chiavi sospette (solo controllo locale .pkl + data_map)",
    )
    args = parser.parse_args()

    print(f"Caricamento dataset da: {args.pkl}")
    with open(args.pkl, "rb") as f:
        data_dict = pickle.load(f)["data"]

    files_by_key, data_map_path = load_datamap_files()
    if data_map_path:
        print(f"Mappa dati caricata da: {data_map_path}")

    results = []
    for key, clist in sorted(data_dict.items()):
        vals = np.array(
            [np.log10(c.integral) for c in clist if getattr(c, "integral", 0) > 0],
            dtype=np.float64,
        )
        if len(vals) < 50:
            continue

        gmm = fit_1d_gmm2(vals)
        files = files_by_key.get(key, [])
        min_w = min(gmm["w1"], gmm["w2"])

        # Criterio di allarme su log10(Integral):
        # 1) Almeno 2 picchi locali reali nell'istogramma, OPPURE
        # 2) Ashman's D >= 1.9 con componente secondaria >= 10% e dBIC > 35, OPPURE
        # 3) Dispersione totale anormalmente larga (std_tot > 0.115 dex) con dmu >= 0.08
        is_bimodal = (
            (gmm["n_peaks"] >= 2)
            or (gmm["ashman_d"] >= 1.9 and min_w >= 0.10 and gmm["delta_bic"] > 35.0)
            or (gmm["std_tot"] > 0.115 and gmm["dmu"] >= 0.08 and min_w >= 0.10)
        )

        results.append(
            {
                "key": key,
                "n": len(vals),
                "vals": vals,
                "gmm": gmm,
                "files": files,
                "is_bimodal": is_bimodal,
            }
        )

    results.sort(key=lambda r: (r["is_bimodal"], r["gmm"]["ashman_d"], r["gmm"]["std_tot"]), reverse=True)

    print("\n" + "=" * 114)
    print(
        f"{'DATA Key (z, P, T, H)':<28} | {'N':>5} | {'Files':>5} | {'std':>5} | "
        f"{'mu1':>5} | {'mu2':>5} | {'GainX':>5} | {'w_min':>5} | {'AshmD':>5} | {'Peaks':>5} | {'Flag'}"
    )
    print("-" * 114)

    flagged = []
    for r in results:
        g = r["gmm"]
        flag_str = "<<< BIMODAL / SHIFT" if r["is_bimodal"] else "OK"
        print(
            f"{str(r['key']):<28} | {r['n']:5d} | {len(r['files']):5d} | {g['std_tot']:5.3f} | "
            f"{g['mu1']:5.3f} | {g['mu2']:5.3f} | {g['gain_ratio']:4.2f}x | "
            f"{min(g['w1'], g['w2']):5.2f} | {g['ashman_d']:5.2f} | {g['n_peaks']:5d} | {flag_str}"
        )
        if r["is_bimodal"]:
            flagged.append(r)

    print("=" * 114)
    print(f"\nChiavi totali analizzate: {len(results)} | Chiavi sospette/bimodali: {len(flagged)}")

    if not flagged:
        print("\nNessuna chiave bimodale rilevata!")
        return

    print("\n=== DETTAGLIO RUN DA CONTROLLARE NEL RUNLOG ===")
    for r in flagged:
        g = r["gmm"]
        print(
            f"\n* Chiave {r['key']}  (N={r['n']}, picchi GMM: {g['mu1']:.3f} vs {g['mu2']:.3f}, "
            f"rapporto guadagno={g['gain_ratio']:.2f}x, pesi={g['w1']:.2f}/{g['w2']:.2f}):"
        )
        r["fstats"] = []
        if not r["files"]:
            print("    (Nessun file associato trovato in data_map)")
            continue

        if not args.no_remote and len(r["files"]) >= 2:
            r["fstats"] = inspect_remote_roots_for_key(r["files"])
            for fs in r["fstats"]:
                if fs["n"] > 0:
                    print(
                        f"    - Run {fs['run']:<7} | N(pre-npix)={fs['n']:4d} | "
                        f"median log10(Int)={fs['median']:.4f} (10^med={10**fs['median']:7.0f}) | "
                        f"{os.path.basename(fs['url'])}"
                    )
                else:
                    print(f"    - Run {fs['run']:<7} | errore lettura: {fs.get('err', '')} | {fs['url']}")
        else:
            for url in r["files"]:
                print(f"    - Run {_extract_run_id(url):<7} | {url}")

    # Plot diagnostico per le sole chiavi flagged
    n_plot = len(flagged)
    ncols = min(3, n_plot)
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False)

    for ax, r in zip(axes.flat, flagged):
        vals = r["vals"]
        bins = np.linspace(min(vals.min(), 3.65), max(vals.max(), 4.45), 65)

        valid_fs = [fs for fs in r.get("fstats", []) if fs.get("vals") is not None and len(fs["vals"]) > 0]
        if len(valid_fs) >= 2:
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                color="black",
                lw=1.5,
                label=f"PKL Totale (N={len(vals)})",
            )
            for fs in valid_fs:
                ax.hist(
                    fs["vals"],
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    label=f"Run {fs['run']} (med={fs['median']:.3f})",
                )
        else:
            ax.hist(vals, bins=bins, density=True, alpha=0.65, color="tab:red", label=f"PKL (N={len(vals)})")

        g = r["gmm"]
        ax.axvline(g["mu1"], color="navy", ls="--", lw=1.1)
        ax.axvline(g["mu2"], color="darkred", ls="--", lw=1.1)
        ax.set_title(
            f"DATA {r['key']}\nGain ratio={g['gain_ratio']:.2f}x | Ashman D={g['ashman_d']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("log10(Integral)")
        ax.legend(fontsize=7.2)
        ax.grid(alpha=0.3)

    for ax in axes.flat[n_plot:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=130)
    print(f"\nPlot diagnostico salvato in: {PLOT_OUT}")


if __name__ == "__main__":
    main()