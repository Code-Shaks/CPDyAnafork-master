#!/usr/bin/env python3
"""
calculations.py – CPDyAna core (v3.6 “Blocks + Brutally Straight”)
=================================================================
• MSD computation            – identical to the original CPDyAna code
• Linear‑window detection    – longest stretch with |d²MSD/dt²| ≤ tol
• Slope (diffusivity)        – plain Δy / Δt over that stretch
• Slope error (s.e.m.)       – _block_stats_slope on that same stretch
Everything else (API & data layout) is unchanged.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple, Optional

# ────────────────────────────────────────────────────────────────
# 1) Original FFT helper (unchanged)
# ────────────────────────────────────────────────────────────────
def calc_fft(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    F = np.fft.fft(x, n=2 * n)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)[:n].real
    return res / (n - np.arange(n))


# ────────────────────────────────────────────────────────────────
# 2) Original per‑ion tracer MSD (unchanged)
# ────────────────────────────────────────────────────────────────
def calc_msd_tracer(r: np.ndarray, idx: np.ndarray):
    n, dim = r.shape
    r_sq = np.square(r)
    r_sq = np.vstack([r_sq, np.zeros((1, dim))])
    rev_cumsum = np.cumsum(r_sq[::-1], axis=0)[::-1] * 2
    lag_len = n - idx
    part1_c = (rev_cumsum[idx].T - r_sq[idx].T) / lag_len
    part1 = part1_c.sum(axis=0)
    part2_c = np.vstack([calc_fft(r[:, k]) for k in range(dim)])
    part2 = part2_c.sum(axis=0)
    msd_scalar = part1 - 2 * part2[idx]
    msd_dim = part1_c - 2 * part2_c[:, idx]
    return msd_scalar, msd_dim


# ────────────────────────────────────────────────────────────────
# 3) Original wrappers & original disp_sum / msd_charged
# ────────────────────────────────────────────────────────────────
def msd_tracer(direction, corrected_structure, step, mobile_mask):
    msd_ions = np.empty([0, len(np.arange(1, step, 1))])
    num_mobile_ions = len(mobile_mask[direction, :])
    structure_shape = (num_mobile_ions, corrected_structure.shape[2], corrected_structure.shape[3])
    structure = np.zeros(structure_shape)
    for X in range(len(structure[:, 0, 0])):
        for Y in range(len(structure[0, :, 0])):
            for Z in range(len(structure[0, 0, :])):
                structure[X, Y, Z] = corrected_structure[direction, X, Y, Z]
    for i in range(num_mobile_ions):
        msd_i, _ = calc_msd_tracer(structure[i, :, :], np.arange(1, step, 1))
        msd_ions = np.append(msd_ions, msd_i.reshape(1, len(np.arange(1, step, 1))), axis=0)
    return np.average(msd_ions, axis=0)


def _disp_sum(delt: int, mobile_pos: np.ndarray) -> float:
    """Original pairwise displacement sum (triple loop, but faithful)."""
    t = np.arange(len(mobile_pos[0, :, 0]) - delt)
    return sum(sum(np.square(sum(abs(mobile_pos[:, t + delt, :] - mobile_pos[:, t, :])))))


def msd_charged(direction, conduction_pos, final_msd, mobile_mask):
    MSD_charged = np.zeros(len(conduction_pos[direction, 0, :, 0]) - 1)
    num_mobile_ions = len(mobile_mask[direction, :])
    structure_shape = (num_mobile_ions, conduction_pos.shape[2], conduction_pos.shape[3])
    structure = np.zeros(structure_shape)
    for X in range(len(structure[:, 0, 0])):
        for Y in range(len(structure[0, :, 0])):
            for Z in range(len(structure[0, 0, :])):
                structure[X, Y, Z] = conduction_pos[direction, X, Y, Z]
    for delt in range(1, len(structure[0, :, 0])):
        MSD_charged[delt - 1] = (
            _disp_sum(delt, structure) - final_msd[delt - 1] * len(structure[:, 0, 0]) * (len(structure[0, :, 0]) - delt)
        ) / (0.5 * len(structure[:, 0, 0]) * (len(structure[:, 0, 0]) - 1) * (len(structure[0, :, 0]) - delt))
    return MSD_charged


# ────────────────────────────────────────────────────────────────
# 4) Statistics helpers
# ────────────────────────────────────────────────────────────────
def _block_stats_slope(t: np.ndarray, y: np.ndarray, block: int) -> Tuple[float, float]:
    """Original block‑statistics mean slope and its s.e.m."""
    if block < 2:
        raise ValueError("block must be ≥2 for block statistics")
    if t.size != y.size:
        t = t[-y.size:]
    nblocks = y.size // block
    if nblocks == 0:
        raise ValueError("block larger than trajectory length")
    cut = nblocks * block
    y_r = y[:cut].reshape(nblocks, block)
    t_r = t[:cut].reshape(nblocks, block)
    slopes = [linregress(t_r[b], y_r[b]).slope for b in range(nblocks)]
    mean = float(np.mean(slopes))
    sem = float(np.std(slopes, ddof=1) / np.sqrt(nblocks))
    return mean, sem


# ────────────────────────────────────────────────────────────────
# Adaptive flat‑window finder (no regression, guaranteed to return)
# ────────────────────────────────────────────────────────────────
def _find_linear_window(
    t: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-3,         # ← back to the original kw‑name
    min_pts: int = 5,
    max_tol: float = 1e-1,
    grow: float = 10.0,
) -> Tuple[int, int]:
    """
    Return start & end indices (inclusive) of the longest region whose
    |d²y/dt²| ≤ tol.  If nothing matches, multiply tol by *grow* until
    we succeed or hit *max_tol*.  Always returns *some* window, so the
    pipeline never crashes.
    """
    if t.size != y.size:
        t = t[-y.size:]

    # --- curvature & initial mask
    with np.errstate(divide='ignore', invalid='ignore'):
        curv = np.abs(np.diff(y, 2) / np.diff(t, 2))
    ok   = np.concatenate([[False], curv <= tol, [False]])

    starts = np.where(np.diff(ok.astype(int)) == 1)[0]
    ends   = np.where(np.diff(ok.astype(int)) == -1)[0]

    # escalate tol until we get a long-enough segment
    current_tol = tol
    while (starts.size == 0 or np.max(ends - starts) + 1 < min_pts) and current_tol < max_tol:
        current_tol *= grow
        ok = np.concatenate([[False], curv <= current_tol, [False]])
        starts = np.where(np.diff(ok.astype(int)) == 1)[0]
        ends   = np.where(np.diff(ok.astype(int)) == -1)[0]

    if starts.size > 0:
        seg_len = ends - starts
        idx     = np.argmax(seg_len)
        return int(starts[idx]), int(ends[idx])       # inclusive

    # last‑ditch: choose flattest single block
    block = max(min_pts, (y.size // 10) or 1)
    blk_slopes = [
        (np.mean(y[i:i+block]), i, i+block-1)
        for i in range(0, y.size - block)
    ]
    _, s, e = min(blk_slopes, key=lambda trip: abs(trip[0]))
    return s, e
# ────────────────────────────────────────────────────────────────
# 5) Public API – calculate_msd (classic MSD + new slope logic)
# ────────────────────────────────────────────────────────────────
def calculate_msd(
    diffusing_elements: List[str],
    diffusivity_direction_choices: List[str],
    diffusivity_type_choices: List[str],
    pos_full: np.ndarray,
    posit: np.ndarray,
    mobile_ion: np.ndarray,
    dt: np.ndarray,
    last: int,
    slope_first: int,
    slope_last: int,
    block: int,
    timestep_fs: float = 1.0,
    verbose: bool = False,
) -> Dict[str, dict]:

    def _suffix_dim(lbl: str):
        return ("", 3) if lbl == "XYZ" else (f"_{lbl}", 2) if lbl in {"XY", "YZ", "ZX"} else (f"_{lbl}", 1)

    if not diffusivity_type_choices:
        return {}

    nstep = posit.shape[2]
    result: Dict[str, dict] = {}

    for elem in diffusing_elements:
        for dir_idx, dir_lbl in enumerate(diffusivity_direction_choices):
            suf, dim = _suffix_dim(dir_lbl)
            if verbose:
                print(f"\n▶ {elem:>3s} | {dir_lbl:<3s}")
            for d_type in diffusivity_type_choices:
                # ---- MSD ----------------------------------------------------
                if d_type == "Tracer":
                    msd_raw = msd_tracer(dir_idx, posit, nstep, mobile_ion)
                elif d_type == "Charged":
                    tracer = msd_tracer(dir_idx, posit, nstep, mobile_ion)
                    msd_raw = msd_charged(dir_idx, posit, tracer, mobile_ion)
                else:
                    raise ValueError(f"unknown type: {d_type}")

                # ---- linear window detection -------------------------------
                t_lags = dt[-msd_raw.size:]
                s_idx, e_idx = _find_linear_window(t_lags, msd_raw, tol=1e-3, min_pts=max(5, block))
                if verbose:
                    print(f"    {d_type}: linear window {s_idx}–{e_idx} (len={e_idx-s_idx+1})")

                # ---- slope by plain rise‑over‑run ---------------------------
                slope = (msd_raw[e_idx] - msd_raw[s_idx]) / (t_lags[e_idx] - t_lags[s_idx])

                # ---- error via block stats on the SAME window --------------
                win_t = t_lags[s_idx:e_idx + 1]
                win_y = msd_raw[s_idx:e_idx + 1]
                # if window shorter than one block → shrink the block
                blk = min(block, max(2, len(win_t) // 2))
                _, slope_sem = _block_stats_slope(win_t, win_y, blk)

                D = (abs(slope) * 1e-4) / (2 * dim)
                D_err = (slope_sem * 1e-4) / (2 * dim)
                bucket = result.setdefault(elem, {})
                bucket.setdefault(f"{d_type}_msd_array{suf}", []).append(msd_raw.tolist())
                bucket.setdefault(f"{d_type}_time_array{suf}", []).append(t_lags.tolist())
                bucket.setdefault(f"{d_type}_diffusivity{suf}", []).append(D)
                bucket.setdefault(f"{d_type}_slope_sem{suf}", []).append(slope_sem)
                bucket.setdefault(f"{d_type}_diffusivity_error{suf}", []).append(D_err)

                if verbose:
                    print(f"    {d_type:<7} D = {D:9.3e} ± {D_err:9.3e} cm²/s")

    return result

# ────────────────────────────────────────────────────────────────
# 6) Public API – compute ionic density
# ────────────────────────────────────────────────────────────────
