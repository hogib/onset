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

**The trigger is ayzek's** (`TriggerRule`, `trigger_tokens`): a rising edge,
or, while p stays at or above the threshold, a restart of dt after it had
grown, which is how an onset inside another event's coda is caught; and no
trigger whose P date is within `min_gap_s` of the last one's. Validation
traces that carry a second onset (`data._second`) are scored on it too, and
model selection counts both onsets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELAYS_S = (0.25, 0.5, 1.0, 2.0, 4.0)
SECOND_DT_WINDOW_S = 3.0      # second onsets: the lowest dt in this long after their P
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


@dataclass
class TriggerRule:
    """ayzek's transformer trigger (src/pipeline/trigger.hpp and the
    processor's minimum time between triggers)."""
    dt_reset: bool = True
    below: float = 2.0          # dt at or below this is a restart ...
    frm: float = 5.0            # ... once dt had reached this since the last trigger
    tokens: int = 2             # ... for this many tokens in a row
    min_gap_s: float = 15.0     # between the P dates (t - dt) of two triggers
    max_dt: float = 10.0

    @classmethod
    def from_config(cls, tcfg, mcfg) -> "TriggerRule":
        return cls(bool(tcfg.dt_reset), tcfg.dt_reset_below, tcfg.dt_reset_from,
                   int(tcfg.dt_reset_tokens), tcfg.min_trigger_gap_s, mcfg.max_dt_s)


def trigger_tokens(p: np.ndarray, dt: np.ndarray, thr: float, release: float,
                   stride: int, fs: float, rule: TriggerRule | None = None) -> np.ndarray:
    """Token indices where the trigger fires. Without `rule`, rising edges
    only (`rising_edges`).

    With it, the stream is cut into runs of p >= release (the trigger re-arms
    between them). In a run the first token at or above `thr` is the rising
    edge; after each trigger at token L, a later token j fires on a restart
    when p[j] >= thr, dt[j] <= below and max(dt[L:j]) >= frm held for
    `tokens` tokens in a row (any token below `thr` breaks the row). Candidates
    closer than `min_gap_s` in P date to the last accepted trigger are
    dropped, as ayzek does, but still restart the dt bookkeeping."""
    if rule is None:
        return rising_edges(p, thr, release)
    active = p >= release
    if not active.any():
        return np.zeros(0, int)
    d = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    cand = []
    for a, b in zip(starts, ends):
        hi = np.flatnonzero(p[a:b] >= thr)
        if not len(hi):
            continue
        L = a + int(hi[0])
        cand.append(L)
        if not rule.dt_reset:
            continue
        while L + 1 < b:
            seg = slice(L + 1, b)
            peak = np.maximum.accumulate(dt[L:b - 1])          # max(dt[L:j]) for j in seg
            ok = (p[seg] >= thr) & (dt[seg] <= rule.below) & (peak >= rule.frm)
            if rule.tokens > 1:
                run = np.convolve(ok.astype(np.int32), np.ones(rule.tokens, np.int32))[:len(ok)]
                ok = run >= rule.tokens
            j = np.flatnonzero(ok)
            if not len(j):
                break
            L = L + 1 + int(j[0])
            cand.append(L)
    cand = np.asarray(cand, int)
    if rule.min_gap_s <= 0 or len(cand) < 2:
        return cand
    t = token_times(len(p), stride, fs)
    dates = t[cand] - np.minimum(dt[cand], rule.max_dt)
    keep, last = [], -np.inf
    for k, date in zip(cand, dates):
        if date - last >= rule.min_gap_s:
            keep.append(k)
            last = date
    return np.asarray(keep, int)


def first_after(edges, t, p_s, tol_s, dt=None):
    """(latency s or nan, onset error s or nan) of the first trigger at or
    after `p_s - tol_s`."""
    hits = edges[t[edges] >= p_s - tol_s] if len(edges) else edges
    if not len(hits):
        return np.nan, np.nan
    j = hits[0]
    return t[j] - p_s, (np.nan if dt is None else (t[j] - dt[j]) - p_s)


def score_event(p, dt, p_s, tol_s, stride, fs, thr, release, rule=None):
    """(latency s or nan, early trigger bool, onset error s or nan)."""
    t = token_times(len(p), stride, fs)
    edges = trigger_tokens(p, dt, thr, release, stride, fs, rule)
    early = bool(len(edges) and t[edges[0]] < p_s - tol_s)
    lat, onset = first_after(edges, t, p_s, tol_s, dt)
    return lat, early, onset


def sweep(events: list[dict], noise: list[dict], stride: int, fs: float,
          thresholds=THRESHOLDS, release_ratio: float = 0.5,
          rule: TriggerRule | None = None) -> list[dict]:
    """One row per threshold.

    Args:
        events: dicts with `p`, `dt` (per-token arrays), `p_s`, `tol_s`, and
            optionally `p2_s`, `tol2_s` for a second onset in the coda.
        noise: dicts with `p`, `dt`, and `missing_tokens` (bool per token) so
            gap time is not counted as monitored time.
        rule: the trigger (`TriggerRule`); None for rising edges only.
    """
    noise_hours = sum((~n["missing_tokens"]).sum() for n in noise) * stride / fs / 3600
    rows = []
    for thr in thresholds:
        release = thr * release_ratio
        lat, early, onset, lat2, dtmin2 = [], [], [], [], []
        for e in events:
            t = token_times(len(e["p"]), stride, fs)
            edges = trigger_tokens(e["p"], e["dt"], thr, release, stride, fs, rule)
            early.append(bool(len(edges) and t[edges[0]] < e["p_s"] - e["tol_s"]))
            l, o = first_after(edges, t, e["p_s"], e["tol_s"], e["dt"])
            lat.append(l)
            onset.append(o)
            if np.isfinite(e.get("p2_s", np.nan)):
                lat2.append(first_after(edges, t, e["p2_s"], e.get("tol2_s", 0.0))[0])
                w2 = (t >= e["p2_s"]) & (t <= e["p2_s"] + SECOND_DT_WINDOW_S)
                if w2.any():
                    dtmin2.append(float(np.min(e["dt"][w2])))
        lat, lat2 = np.asarray(lat), np.asarray(lat2)
        fa = sum(len(trigger_tokens(n["p"], n.get("dt", np.zeros_like(n["p"])), thr, release,
                                    stride, fs, rule)) for n in noise)
        row = {"threshold": float(thr),
               "false_per_hour": fa / noise_hours if noise_hours else np.nan,
               "early_rate": float(np.mean(early)) if events else np.nan,
               "detected": float(np.mean(np.isfinite(lat))) if events else np.nan,
               "latency_p50_s": float(np.nanmedian(lat)) if np.isfinite(lat).any() else np.nan,
               "onset_abs_err_p50_s": float(np.nanmedian(np.abs(onset)))
               if np.isfinite(onset).any() else np.nan}
        for d in DELAYS_S:
            row[f"recall@{d}s"] = float(np.mean(lat <= d)) if events else np.nan
            row[f"second_recall@{d}s"] = float(np.mean(lat2 <= d)) if len(lat2) else np.nan
        row["second_n"] = len(lat2)
        # How far dt comes down at a second onset: the trigger needs 2 s.
        row["second_dt_min_p50"] = float(np.median(dtmin2)) if dtmin2 else np.nan
        both = np.concatenate([lat, lat2])
        row["onset_recall@1.0s"] = float(np.mean(both <= 1.0)) if len(both) else np.nan
        rows.append(row)
    return rows


def operating_point(rows: list[dict], fa_target_per_hour: float) -> dict:
    """The lowest threshold whose false-trigger rate is within budget. The
    sweep is monotone in practice (a higher threshold fires less), so the
    lowest admissible threshold is the fastest one."""
    ok = [r for r in rows if r["false_per_hour"] <= fa_target_per_hour]
    return min(ok, key=lambda r: r["threshold"]) if ok else max(rows, key=lambda r: r["threshold"])


def false_per_hour(noise: list[dict], thr: float, release: float, stride: int, fs: float,
                   rule: TriggerRule | None = None) -> float:
    hours = sum((~n["missing_tokens"]).sum() for n in noise) * stride / fs / 3600
    fa = sum(len(trigger_tokens(n["p"], n.get("dt", np.zeros_like(n["p"])), thr, release,
                                stride, fs, rule)) for n in noise)
    return fa / hours if hours else np.nan


def exact_operating_point(rows: list[dict], events: list[dict], noise: list[dict],
                          stride: int, fs: float, fa_target_per_hour: float,
                          rule: TriggerRule | None = None, release_ratio: float = 0.5,
                          iters: int = 14) -> dict:
    """The lowest threshold within the false-trigger budget, found by
    bisection between the grid's last threshold over budget and its first
    within it, instead of the grid point itself.

    A trained model's outputs crowd against 1, where one grid step moves
    recall by tens of points: on the wide KO set, recall within 1 s swung
    between 0.28 and 0.66 from epoch to epoch as the operating point jumped
    between neighbouring grid thresholds. The bisection runs in logit, on the
    noise alone (the false-trigger rate falls as the threshold rises), and
    the events are scored once, at the threshold it settles on.
    """
    grid = operating_point(rows, fa_target_per_hour)
    over = [r for r in rows if r["threshold"] < grid["threshold"]
            and not r["false_per_hour"] <= fa_target_per_hour]
    if not over or not noise or not grid["false_per_hour"] <= fa_target_per_hour:
        return grid
    lo = np.log(max(over, key=lambda r: r["threshold"])["threshold"])
    lo = float(lo - np.log1p(-np.exp(lo)))                       # logit
    t = grid["threshold"]
    hi = float(np.log(t) - np.log1p(-t))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        thr = 1.0 / (1.0 + np.exp(-mid))
        if false_per_hour(noise, thr, thr * release_ratio, stride, fs, rule) <= fa_target_per_hour:
            hi = mid
        else:
            lo = mid
    thr = 1.0 / (1.0 + np.exp(-hi))
    return sweep(events, noise, stride, fs, [thr], release_ratio, rule)[0]


def summary(rows: list[dict], fa_target_per_hour: float, op: dict | None = None) -> dict:
    """The operating point's row plus the model-selection score; `op` is the
    exact operating point when one was computed, else the grid's."""
    op = op if op is not None else operating_point(rows, fa_target_per_hour)
    return {"fa_target_per_hour": fa_target_per_hour, **op,
            "score": op.get("onset_recall@1.0s", op["recall@1.0s"])}


# --- geometry ---------------------------------------------------------------

GEO_BINS_S = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 30.0))


def geometry_table(events: list[dict], stride: int, fs: float) -> list[dict]:
    """Distance error by time since P, and before/after S.

    Each event contributes its median over the tokens in a bin, so a long
    trace does not outweigh a short one. `cal_1sd` is the share of events
    whose true log distance lies within the model's own one-sigma: about 0.68
    when the stated uncertainty is honest, lower when it is overconfident.
    """
    rows = []
    bins = [(f"{a:g}-{b:g} s after P", a, b, None) for a, b in GEO_BINS_S]
    bins += [("after P, before S", 0.0, 1e9, "before"), ("after S", 0.0, 1e9, "after")]
    for name, lo, hi, phase in bins:
        d_err, d_rel, cal = [], [], []
        for e in events:
            if "log_dist" not in e or not np.isfinite(e.get("dist_km", np.nan)):
                continue
            t = token_times(len(e["p"]), stride, fs)
            since = t - e["p_s"]
            m = (since >= lo) & (since < hi) & ~e["missing_tokens"]
            if np.isfinite(e.get("p2_s", np.nan)):          # the targets are the first event's
                m &= t < e["p2_s"] - e.get("tol2_s", 0.0)
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
        if d_err:
            rows.append({"window": name, "n": len(d_err),
                         "dist_abs_err_km_p50": float(np.median(d_err)),
                         "dist_rel_err_p50": float(np.expm1(np.median(d_rel))),
                         "cal_1sd": float(np.mean(cal))})
    return rows
