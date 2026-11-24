import os
from collections import deque
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    max_grow_radius=5,
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


def clean_single_event_map(
    ev_sum,
    ev_occ,
    topo_veto_mask=None,
    interior_mask=None,
    nsigma=25.0,
    skirt_nsigma=15.0,
    max_radius=5,
    hard_cap=400.0,
    dilate=1,
    max_veto_bins=800,
):
    """
    Pulisce la mappa 2D di un singolo evento (ev_sum, ev_occ) prima dell'accumulo
    in int2d_images / occ2d_images.

    Restituisce:
        ev_sum_clean, ev_occ_clean, veto_mask, full_event_rejected, info_dict
    """
    active = (ev_occ > 0) & np.isfinite(ev_sum)
    if topo_veto_mask is not None:
        active &= (~topo_veto_mask)

    empty_info = {"core_thr": np.nan, "max_before": 0.0, "max_after": 0.0, "n_vetoed": 0}
    if not np.any(active):
        return ev_sum, ev_occ, np.zeros_like(active, dtype=bool), False, empty_info

    ev_prof = np.zeros_like(ev_sum, dtype=np.float64)
    ev_prof[active] = ev_sum[active] / ev_occ[active]
    max_before = float(np.max(ev_prof[active]))

    stats_mask = active & interior_mask if interior_mask is not None else active
    if not np.any(stats_mask):
        stats_mask = active

    vals = ev_prof[stats_mask]
    median_val = float(np.median(vals))
    mad_val = float(np.median(np.abs(vals - median_val)))
    robust_std = 1.4826 * mad_val
    if robust_std <= 0:
        q75, q25 = np.percentile(vals, [75, 25])
        robust_std = float((q75 - q25) / 1.349) if q75 > q25 else float(np.std(vals))

    core_thr = median_val + float(nsigma) * robust_std
    base_skirt_thr = median_val + float(skirt_nsigma) * robust_std

    has_hard_cap = hard_cap is not None and float(hard_cap) > 0
    if has_hard_cap:
        core_thr = min(core_thr, float(hard_cap))
        base_skirt_thr = min(base_skirt_thr, core_thr)

    # Fast-path: nessun bin sopra core_thr -> nessuna scarica in questo evento
    if not np.any(active & (ev_prof > core_thr)):
        info = {"core_thr": core_thr, "max_before": max_before, "max_after": max_before, "n_vetoed": 0}
        return ev_sum, ev_occ, np.zeros_like(active, dtype=bool), False, info

    dyn_mask = detect_bounded_spark_mask(
        prof=ev_prof,
        base_valid=active,
        core_thr=core_thr,
        base_skirt_thr=base_skirt_thr,
        peak_fraction=0.02,
        max_grow_radius=max_radius,
        dilate_bins=dilate,
    )

    if has_hard_cap:
        hard_hits = active & (ev_prof > float(hard_cap))
        if np.any(hard_hits):
            dyn_mask |= binary_dilate_2d(hard_hits, radius=dilate) & active

    n_vetoed = int(np.sum(dyn_mask))
    if n_vetoed == 0:
        info = {"core_thr": core_thr, "max_before": max_before, "max_after": max_before, "n_vetoed": 0}
        return ev_sum, ev_occ, dyn_mask, False, info

    if max_veto_bins is not None and n_vetoed > int(max_veto_bins):
        info = {"core_thr": core_thr, "max_before": max_before, "max_after": 0.0, "n_vetoed": n_vetoed}
        return np.zeros_like(ev_sum), np.zeros_like(ev_occ), dyn_mask, True, info

    ev_sum_clean = ev_sum.copy()
    ev_occ_clean = ev_occ.copy()
    ev_sum_clean[dyn_mask] = 0.0
    ev_occ_clean[dyn_mask] = 0.0

    rem_active = active & (~dyn_mask)
    max_after = float(np.max(ev_prof[rem_active])) if np.any(rem_active) else 0.0
    info = {"core_thr": core_thr, "max_before": max_before, "max_after": max_after, "n_vetoed": n_vetoed}
    return ev_sum_clean, ev_occ_clean, dyn_mask, False, info


def save_event_spark_check(
    event_label,
    ev_sum_before,
    ev_occ_before,
    veto_mask,
    x_edges,
    y_edges,
    threshold,
    check_dir,
):
    os.makedirs(check_dir, exist_ok=True)
    out_png = os.path.join(check_dir, f"ev_spark_{event_label}.png")

    active_before = ev_occ_before > 0
    prof_before = np.full_like(ev_sum_before, np.nan, dtype=np.float64)
    prof_before[active_before] = ev_sum_before[active_before] / ev_occ_before[active_before]

    active_after = active_before & (~veto_mask)
    prof_after = np.where(active_after, prof_before, np.nan)

    max_before = float(np.nanmax(prof_before)) if np.any(active_before) else 0.0
    max_after = float(np.nanmax(prof_after)) if np.any(active_after) else 0.0
    n_removed = int(np.sum(veto_mask))

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5), dpi=150)

    m0 = axes[0].pcolormesh(x_edges, y_edges, prof_before.T, shading="flat", cmap="rainbow")
    cb0 = fig.colorbar(m0, ax=axes[0], pad=0.02, fraction=0.046)
    cb0.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[0].set_title(f"1. Evento grezzo (max={max_before:.0f} ADC)", fontsize=11, fontweight="bold")

    axes[1].pcolormesh(x_edges, y_edges, prof_after.T, shading="flat", cmap="Greys", alpha=0.55)
    axes[1].pcolormesh(
        x_edges, y_edges, np.where(veto_mask, 1.0, np.nan).T,
        shading="flat", cmap="autumn", vmin=0, vmax=1, alpha=0.95
    )
    axes[1].set_title(f"2. Bin rimossi ({n_removed} | thr={threshold:.1f})", fontsize=11, fontweight="bold")

    m2 = axes[2].pcolormesh(x_edges, y_edges, prof_after.T, shading="flat", cmap="rainbow")
    cb2 = fig.colorbar(m2, ax=axes[2], pad=0.02, fraction=0.046)
    cb2.set_label("Mean Charge / Hit [ADC]", fontsize=10)
    axes[2].set_title(f"3. Evento pulito (max={max_after:.0f} ADC)", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("X [pixel]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3, color="gray")
    axes[0].set_ylabel("Y [pixel]", fontsize=10)

    fig.suptitle(f"Pulizia scarica singolo evento — {event_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)