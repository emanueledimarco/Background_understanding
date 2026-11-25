import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

PKL_IN = "data/cache/cygno_clusters_dataset.pkl"
PKL_OUT = "data/cache/cygno_clusters_dataset_nopileup.pkl"
PLOT_OUT = "check_pileup_cleaning.png"

LOG10_2 = np.log10(2.0)  # 0.30103
LOG10_3 = np.log10(3.0)  # 0.47712


def _get_scalar(c, name, fallback_idx=None):
    val = getattr(c, name, None)
    if (
        val is None
        and fallback_idx is not None
        and hasattr(c, "scalars")
        and c.scalars is not None
    ):
        arr = np.asarray(c.scalars).ravel()
        if len(arr) > fallback_idx:
            val = arr[fallback_idx]
    return float(val) if val is not None else None


def _elongation_score(c):
    """Punteggio di allungamento: più alto = più probabile doppio cluster fuso."""
    l = _get_scalar(c, "length")
    s = _get_scalar(c, "slimness")
    w = _get_scalar(c, "width")
    if s is None and w is not None and l is not None and l > 0:
        s = w / l
    if l is not None and s is not None and np.isfinite(l) and np.isfinite(s):
        return l / max(s, 0.15)
    if l is not None and np.isfinite(l):
        return l
    if s is not None and np.isfinite(s):
        return 1.0 / max(s, 0.15)
    return 0.0


def _smooth_1d(y, sigma_bins=2.0):
    radius = int(max(3, round(3 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def _gauss(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def detect_and_fit_2x_pileup(log_int):
    """
    Assume che il picco dominante (bulk) sia 1xE no-pileup e verifica se esiste
    un picco/eccesso stretto centrato a ~2xE (mu1 + log10(2) +- 0.04 dex).
    """
    lo, hi = np.percentile(log_int, [0.2, 99.8])
    hi_plot = max(hi + 0.05, lo + 0.80)
    bins = np.linspace(lo - 0.05, hi_plot, 110)
    hist, edges = np.histogram(log_int, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = _smooth_1d(hist.astype(np.float64), sigma_bins=1.8)

    # 1. Picco dominante (bulk 1xE) = massimo globale
    peak_idx = int(np.argmax(smooth))
    mu1_0 = float(centers[peak_idx])
    amp1_0 = float(max(smooth[peak_idx], 1.0))

    # 2. Larghezza del lato destro del bulk dalla prima discesa a metà altezza (HWHM_R)
    half_h = 0.5 * amp1_0
    sigma_r0 = 0.075
    for j in range(peak_idx + 1, len(smooth)):
        if centers[j] > mu1_0 + 0.22:
            break
        if smooth[j] <= half_h:
            # Interpolazione lineare della posizione di metà altezza
            y0, y1 = smooth[j - 1], smooth[j]
            frac = (y0 - half_h) / max(y0 - y1, 1e-9)
            x_half = centers[j - 1] + frac * (centers[j] - centers[j - 1])
            sigma_r0 = (x_half - mu1_0) / np.sqrt(2.0 * np.log(2.0))
            break
    sigma_r0 = float(np.clip(sigma_r0, 0.030, 0.160))

    # Fit solo dal picco del bulk verso destra (immune da asimmetria sinistra / soglia 5000)
    fit_mask = (centers >= mu1_0 - 0.015) & (centers <= mu1_0 + 0.65)
    xf = centers[fit_mask]
    yf = hist[fit_mask].astype(np.float64)
    err = np.sqrt(np.maximum(yf, 1.0))

    # --- Modello H0: SOLO BULK (nessun pile-up) ---
    def res_h0(p):
        a1, mu1, sr = p
        return (_gauss(xf, a1, mu1, sr) - yf) / err

    h0 = least_squares(
        res_h0,
        x0=[amp1_0, mu1_0, sigma_r0],
        bounds=(
            [0.5 * amp1_0, mu1_0 - 0.015, 0.70 * sigma_r0],
            [1.4 * amp1_0, mu1_0 + 0.015, 1.30 * sigma_r0],
        ),
    )
    a1_0, mu1_h0, sr_h0 = h0.x
    y_h0 = _gauss(xf, a1_0, mu1_h0, sr_h0)

    # --- Modello H1: BULK + PICCO STRETTO A 2xE (+ eventuale 3xE) ---
    s2_max = min(0.75 * sigma_r0, 0.065)
    s2_init = min(0.68 * sigma_r0, 0.050)
    s2_min = min(0.022, 0.5 * s2_max)

    def eval_h1(p, x):
        a1, mu1, sr, a2, mu2, s2, a3 = p
        mu3 = mu1 + LOG10_3 + (mu2 - (mu1 + LOG10_2))
        y1 = _gauss(x, a1, mu1, sr)
        y2 = _gauss(x, a2, mu2, s2)
        y3 = _gauss(x, a3, mu3, s2)
        return y1, y2, y3

    def res_h1(p):
        y1, y2, y3 = eval_h1(p, xf)
        return ((y1 + y2 + y3) - yf) / err

    # Stima iniziale di A2 attorno a mu1 + log10(2)
    win2_init = (xf >= mu1_0 + 0.26) & (xf <= mu1_0 + 0.34)
    a2_init = float(max(yf[win2_init].max(), 1.0)) if np.any(win2_init) else 1.0
    a2_init = min(a2_init, 0.8 * amp1_0)

    h1 = least_squares(
        res_h1,
        x0=[a1_0, mu1_h0, sr_h0, a2_init, mu1_h0 + LOG10_2, s2_init, 0.05 * a2_init],
        bounds=(
            [0.5 * amp1_0, mu1_0 - 0.015, 0.70 * sigma_r0, 0.0, mu1_0 + 0.26, s2_min, 0.0],
            [1.4 * amp1_0, mu1_0 + 0.015, 1.30 * sigma_r0, 1.0 * amp1_0, mu1_0 + 0.34, s2_max, 0.35 * amp1_0],
        ),
    )
    a1, mu1, sr, a2, mu2, s2, a3 = h1.x
    y1_f, y2_f, y3_f = eval_h1(h1.x, xf)

    # 3. Test di significatività del picco 2xE nella sua finestra [mu2 - 2*s2, mu2 + 2*s2]
    w2 = (xf >= mu2 - 2.0 * s2) & (xf <= mu2 + 2.0 * s2)
    chi2_h0_w2 = float(np.sum(((y_h0[w2] - yf[w2]) / err[w2]) ** 2))
    chi2_h1_w2 = float(np.sum((((y1_f + y2_f + y3_f)[w2] - yf[w2]) / err[w2]) ** 2))
    delta_chi2 = chi2_h0_w2 - chi2_h1_w2

    obs_exc = float(np.sum(yf[w2] - y_h0[w2]))
    bkg_h0 = float(max(np.sum(y_h0[w2]), 1.0))
    z_sig = obs_exc / np.sqrt(bkg_h0 + max(obs_exc, 0.0))
    n_fit_2x = float(np.sum(y2_f))

    # Accetta il picco 2xE solo se è un vero eccesso localizzato sopra il bulk
    has_pileup = (
        (a2 / max(a1, 1.0) >= 0.025)
        and (obs_exc >= 25.0)
        and (n_fit_2x >= 25.0)
        and (z_sig >= 3.5)
        and (delta_chi2 >= 12.0)
    )

    if not has_pileup:
        return {
            "has_pileup": False,
            "mu1": mu1_h0,
            "sigma_r": sr_h0,
            "a1": a1_0,
            "a2": 0.0,
            "mu2": mu1_h0 + LOG10_2,
            "sigma2": s2_init,
            "a3": 0.0,
            "bin_w": float(edges[1] - edges[0]),
            "z_sig": z_sig,
            "delta_chi2": delta_chi2,
        }

    return {
        "has_pileup": True,
        "mu1": mu1,
        "sigma_r": sr,
        "a1": a1,
        "a2": a2,
        "mu2": mu2,
        "sigma2": s2,
        "a3": a3,
        "bin_w": float(edges[1] - edges[0]),
        "z_sig": z_sig,
        "delta_chi2": delta_chi2,
    }


def pileup_probability(x, p):
    """Probabilità di essere pile-up (2xE / 3xE). Sempre 0 sul bulk."""
    x = np.asarray(x, dtype=np.float64)
    if not p["has_pileup"]:
        return np.zeros_like(x)

    mu1, sr, a1 = p["mu1"], p["sigma_r"], p["a1"]
    mu2, s2, a2, a3 = p["mu2"], p["sigma2"], p["a2"], p["a3"]
    mu3 = mu1 + LOG10_3 + (mu2 - (mu1 + LOG10_2))

    y1 = _gauss(x, a1, mu1, sr)
    y2 = _gauss(x, a2, mu2, s2)
    y3 = _gauss(x, a3, mu3, s2)

    p_pu = (y2 + y3) / (y1 + y2 + y3 + 1e-15)

    # Protezione rigida del bulk: mai toccare sotto max(mu1 + 0.14, mu2 - 2.5*s2)
    safe_bulk_edge = max(mu1 + 0.14, mu2 - 2.5 * s2)
    p_pu = np.where(x < safe_bulk_edge, 0.0, p_pu)

    # Oltre mu2 + 2.5*s2 (zona 3xE / pile-up multiplo) rimuovi se c'è pile-up
    p_pu = np.where(x > mu2 + 2.5 * s2, np.maximum(p_pu, 0.95), p_pu)
    return np.clip(p_pu, 0.0, 1.0)


def filter_key_2x_pileup(valid_pairs, params, seed=42):
    """
    Mantiene il 100% del bulk; nella sola zona del picco 2xE (se rilevato)
    rimuove la frazione stimata di pile-up scegliendo i cluster più allungati.
    """
    if not params["has_pileup"]:
        return [c for c, _ in valid_pairs]

    rng = np.random.default_rng(seed)
    log_ints = np.array([li for _, li in valid_pairs], dtype=np.float64)
    p_keep = 1.0 - pileup_probability(log_ints, params)

    kept = []
    overlap_indices = []
    for i, (c, _) in enumerate(valid_pairs):
        if p_keep[i] >= 0.999:
            kept.append(c)
        elif p_keep[i] > 1e-3:
            overlap_indices.append(i)

    if not overlap_indices:
        return kept

    overlap_indices = np.array(overlap_indices, dtype=np.int64)
    o_li = log_ints[overlap_indices]

    bin_w = 0.015
    edges = np.arange(o_li.min(), o_li.max() + bin_w + 1e-6, bin_w)
    bin_ids = np.digitize(o_li, edges) - 1

    for b in np.unique(bin_ids):
        in_bin = overlap_indices[bin_ids == b]
        if len(in_bin) == 0:
            continue

        expected_keep = float(p_keep[in_bin].sum())
        n_keep = int(np.floor(expected_keep))
        if rng.random() < (expected_keep - n_keep):
            n_keep += 1
        n_keep = min(max(n_keep, 0), len(in_bin))

        if n_keep == 0:
            continue
        if n_keep == len(in_bin):
            for idx in in_bin:
                kept.append(valid_pairs[idx][0])
            continue

        scores = np.array(
            [_elongation_score(valid_pairs[idx][0]) for idx in in_bin],
            dtype=np.float64,
        )
        scale = max(np.std(scores), 1e-6)
        noisy_scores = scores + 0.10 * scale * rng.standard_normal(len(scores))
        order = np.argsort(noisy_scores)
        for idx in in_bin[order[:n_keep]]:
            kept.append(valid_pairs[idx][0])

    return kept


def main():
    with open(PKL_IN, "rb") as f:
        payload = pickle.load(f)

    data_dict = payload["data"]
    sim_dict = payload["sim"]

    clean_data = {}
    diag_examples = {}

    tot_d_in, tot_d_out = 0, 0
    print(
        f"{'DATA Key (z, P, T, H)':<30} | {'In':>6} | {'Out':>6} | {'Kept':>6} | "
        f"{'mu1':>5} | {'sigR':>5} | {'mu2':>5} | {'sig2':>5} | {'Status'}"
    )
    print("-" * 102)

    for key_idx, (key, clist) in enumerate(sorted(data_dict.items())):
        tot_d_in += len(clist)
        valid_pairs = []
        for c in clist:
            integ = _get_scalar(c, "integral", fallback_idx=0)
            if integ is not None and integ > 0:
                valid_pairs.append((c, np.log10(integ)))

        if len(valid_pairs) < 80:
            clean_data[key] = [c for c, _ in valid_pairs]
            tot_d_out += len(clean_data[key])
            continue

        log_ints = np.array([li for _, li in valid_pairs], dtype=np.float64)
        params = detect_and_fit_2x_pileup(log_ints)
        kept = filter_key_2x_pileup(valid_pairs, params, seed=42 + key_idx)

        clean_data[key] = kept
        tot_d_out += len(kept)

        pct = 100.0 * len(kept) / max(len(clist), 1)
        status = (
            f"2xE DETECTED (Z={params['z_sig']:.1f})"
            if params["has_pileup"]
            else "NO 2xE (bulk 100%)"
        )
        print(
            f"{str(key):<30} | {len(clist):6d} | {len(kept):6d} | {pct:5.1f}% | "
            f"{params['mu1']:5.3f} | {params['sigma_r']:5.3f} | "
            f"{params['mu2']:5.3f} | {params['sigma2']:5.3f} | {status}"
        )

        z_val = key[0]
        if key in (
            (25.0, 0.916, 21.4, 1.1),
            (25.0, 0.903, 21.8, 1.3),
        ) or z_val not in {k[0] for k in diag_examples}:
            kept_li = np.array(
                [
                    np.log10(_get_scalar(c, "integral", fallback_idx=0))
                    for c in kept
                ]
            )
            diag_examples[key] = (log_ints, kept_li, params)

    print("-" * 102)
    print(
        f"DATA totali: {tot_d_in} -> {tot_d_out} ({100.0 * tot_d_out / max(tot_d_in, 1):.1f}% mantenuti)"
    )
    print("SIM  totali: invariata")

    payload["data"] = clean_data
    payload["sim"] = sim_dict
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        payload["metadata"].setdefault("stats", {})
        payload["metadata"]["stats"]["n_clusters_data"] = tot_d_out
        payload["metadata"]["pileup_cleaning"] = {
            "method": "bulk_right_hwhm_plus_2xE_peak_detection",
        }

    os.makedirs(os.path.dirname(os.path.abspath(PKL_OUT)), exist_ok=True)
    with open(PKL_OUT, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"\nSalvato dataset pulito in: {PKL_OUT}")

    # Plot diagnostico prima / dopo
    keys_to_plot = sorted(diag_examples.keys())[:6]
    ncols = min(3, len(keys_to_plot))
    nrows = (len(keys_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 4.0 * nrows), squeeze=False
    )
    for ax, key in zip(axes.flat, keys_to_plot):
        raw_li, kept_li, p = diag_examples[key]
        bins = np.linspace(min(raw_li.min(), 3.5), max(raw_li.max(), 4.6), 75)
        scale_bins = (bins[1] - bins[0]) / p["bin_w"]

        ax.hist(
            raw_li,
            bins=bins,
            alpha=0.35,
            color="tab:red",
            label=f"Raw (N={len(raw_li)})",
        )
        ax.hist(
            kept_li,
            bins=bins,
            alpha=0.65,
            color="tab:blue",
            label=f"Clean (N={len(kept_li)})",
        )

        x_r = np.linspace(p["mu1"], bins[-1], 250)
        y1 = scale_bins * _gauss(x_r, p["a1"], p["mu1"], p["sigma_r"])
        ax.plot(
            x_r,
            y1,
            "k--",
            lw=1.4,
            label=f"Bulk R ($\\sigma_R$={p['sigma_r']:.3f})",
        )
        if p["has_pileup"]:
            y2 = scale_bins * _gauss(x_r, p["a2"], p["mu2"], p["sigma2"])
            ax.plot(
                x_r,
                y2,
                "g:",
                lw=1.8,
                label=f"2xE ($\\mu_2$={p['mu2']:.2f}, $\\sigma_2$={p['sigma2']:.3f})",
            )
            safe_edge = max(p["mu1"] + 0.14, p["mu2"] - 2.5 * p["sigma2"])
            ax.axvline(safe_edge, color="gray", ls="-.", lw=1.0, alpha=0.8)

        tag = "2xE removed" if p["has_pileup"] else "No 2xE (100% kept)"
        ax.set_title(f"DATA {key} — {tag}", fontsize=8.8)
        ax.set_xlabel("log10(Integral)")
        ax.legend(fontsize=7.2)
        ax.grid(alpha=0.35)

    for ax in axes.flat[len(keys_to_plot) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=130)
    print(f"Plot diagnostico salvato in: {PLOT_OUT}")


if __name__ == "__main__":
    main()