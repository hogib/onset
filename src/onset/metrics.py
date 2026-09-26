"""Detection latency and false triggers: the numbers this detector is judged on.

AUC is not among them. A detector for a stream is judged on two things:

- **How soon after P it fires** on an event: recall within 0.25, 0.5, 1, 2
  and 4 s of the labelled P.
- **How often it fires on noise**: rising edges per hour of noise, with
  hysteresis (it re-arms only after dropping below `release`).

Both depend on the threshold, so they are swept together and reported at the
threshold that meets a false-trigger budget (`fa_target_per_hour`).

The trigger time of token j is the time of its last sample, `(stride*j +
stride - 1) / fs`: the earliest moment a real-time system could know the
token's output. A crossing earlier than `P - tolerance` on an event trace
is a false trigger on that trace (`early`), and it does not count as a
detection. The trace can still detect afterwards, once it has re-armed.
"""
from __future__ import annotations

import numpy as np

DELAYS_S = (0.25, 0.5, 1.0, 2.0, 4.0)
# Even steps in logit, not probability. Without label smoothing a trained
# model's outputs crowd against 1, and a grid in probability steps from 0.990
# to 0.995 in one move -- measured, recall within 1 s fell from 46% to 0.3%
# across that single step.
THRESHOLDS = 1.0 / (1.0 + np.exp(-np.arange(-2.0, 12.01, 0.25)))


def rising_edges(p: np.ndarray, thr: float, release: float) -> np.ndarray:
    """Token indices where p crosses `thr` while armed. Disarms on a crossing,
    re-arms when p falls below `release`."""
    # Vectorised hysteresis. With release < thr, "above" and "below" never
    # coincide, so the detector is armed at j exactly when the latest of them
    # before j was a "below" (or neither has happened yet).
    above, below = p >= thr, p < release
    idx = np.arange(len(p))
    last_above = np.maximum.accumulate(np.where(above, idx, -1))
    last_below = np.maximum.accumulate(np.where(below, idx, -1))
    prev_above = np.concatenate([[-1], last_above[:-1]])
    prev_below = np.concatenate([[-1], last_below[:-1]])
    armed = (prev_above < 0) | (prev_below > prev_above)
    return np.flatnonzero(above & armed)


def token_times(n_tokens: int, stride: int, fs: float) -> np.ndarray:
    return (np.arange(n_tokens) * stride + stride - 1) / fs


def score_event(p, dt, p_s, tol_s, stride, fs, thr, release):
    """(latency s or nan, early trigger bool, onset error s or nan)."""
    t = token_times(len(p), stride, fs)
    edges = rising_edges(p, thr, release)
    early = bool(len(edges) and t[edges[0]] < p_s - tol_s)
    hits = edges[t[edges] >= p_s - tol_s] if len(edges) else edges
    if not len(hits):
        return np.nan, early, np.nan
    j = hits[0]
    return t[j] - p_s, early, (t[j] - dt[j]) - p_s


def sweep(events: list[dict], noise: list[dict], stride: int, fs: float,
          thresholds=THRESHOLDS, release_ratio: float = 0.5) -> list[dict]:
    """One row per threshold.

    Args:
        events: dicts with `p`, `dt` (per-token arrays), `p_s`, `tol_s`.
        noise: dicts with `p`, and `missing_tokens` (bool per token) so gap
            time is not counted as monitored time.
    """
    noise_hours = sum((~n["missing_tokens"]).sum() for n in noise) * stride / fs / 3600
    rows = []
    for thr in thresholds:
        release = thr * release_ratio
        lat, early, onset = [], [], []
        for e in events:
            l, er, o = score_event(e["p"], e["dt"], e["p_s"], e["tol_s"], stride, fs, thr, release)
            lat.append(l)
            early.append(er)
            onset.append(o)
        lat = np.asarray(lat)
        fa = sum(len(rising_edges(n["p"], thr, release)) for n in noise)
        row = {"threshold": float(thr),
               "false_per_hour": fa / noise_hours if noise_hours else np.nan,
               "early_rate": float(np.mean(early)) if events else np.nan,
               "detected": float(np.mean(np.isfinite(lat))) if events else np.nan,
               "latency_p50_s": float(np.nanmedian(lat)) if np.isfinite(lat).any() else np.nan,
               "onset_abs_err_p50_s": float(np.nanmedian(np.abs(onset)))
               if np.isfinite(onset).any() else np.nan}
        for d in DELAYS_S:
            row[f"recall@{d}s"] = float(np.mean(lat <= d)) if events else np.nan
        rows.append(row)
    return rows


def operating_point(rows: list[dict], fa_target_per_hour: float) -> dict:
    """The lowest threshold whose false-trigger rate is within budget. The
    sweep is monotone in practice (a higher threshold fires less), so the
    lowest admissible threshold is the fastest one."""
    ok = [r for r in rows if r["false_per_hour"] <= fa_target_per_hour]
    return min(ok, key=lambda r: r["threshold"]) if ok else max(rows, key=lambda r: r["threshold"])


def summary(rows: list[dict], fa_target_per_hour: float) -> dict:
    op = operating_point(rows, fa_target_per_hour)
    return {"fa_target_per_hour": fa_target_per_hour, **op,
            "score": op["recall@1.0s"]}


# --- geometry ---------------------------------------------------------------

GEO_BINS_S = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 30.0))


def _angle_deg(a):
    return np.degrees(np.abs((a + np.pi) % (2 * np.pi) - np.pi))


def geometry_table(events: list[dict], stride: int, fs: float) -> list[dict]:
    """Distance and back-azimuth error by time since P, and before/after S.

    Each event contributes its median over the tokens in a bin, so a long
    trace does not outweigh a short one. `cal_1sd` is the share of events
    whose true log distance lies within the model's own one-sigma: about 0.68
    when the stated uncertainty is honest, lower when it is overconfident.
    """
    rows = []
    bins = [(f"{a:g}-{b:g} s after P", a, b, None) for a, b in GEO_BINS_S]
    bins += [("after P, before S", 0.0, 1e9, "before"), ("after S", 0.0, 1e9, "after")]
    for name, lo, hi, phase in bins:
        d_err, d_rel, cal, b_err = [], [], [], []
        for e in events:
            if "log_dist" not in e or not np.isfinite(e.get("dist_km", np.nan)):
                continue
            t = token_times(len(e["p"]), stride, fs)
            since = t - e["p_s"]
            m = (since >= lo) & (since < hi) & ~e["missing_tokens"]
            if phase is not None:
                if not np.isfinite(e.get("s_s", np.nan)):
                    continue
                m &= (t < e["s_s"]) if phase == "before" else (t >= e["s_s"] + 0.5)
            if not m.any():
                continue
            y = np.log(max(e["dist_km"], 1.0))
            err = e["log_dist"][m] - y
            d_err.append(np.median(np.abs(np.exp(e["log_dist"][m]) - e["dist_km"])))
            d_rel.append(np.median(np.abs(err)))
            cal.append(np.median(np.abs(err) <= np.exp(0.5 * e["log_dist_var"][m])))
            if np.isfinite(e.get("baz_rad", np.nan)):
                b_err.append(np.median(_angle_deg(e["baz"][m] - e["baz_rad"])))
        if d_err:
            rows.append({"window": name, "n": len(d_err),
                         "dist_abs_err_km_p50": float(np.median(d_err)),
                         "dist_rel_err_p50": float(np.expm1(np.median(d_rel))),
                         "cal_1sd": float(np.mean(cal)),
                         "baz_err_deg_p50": float(np.median(b_err)) if b_err else np.nan,
                         "n_baz": len(b_err)})
    return rows
