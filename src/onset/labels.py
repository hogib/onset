"""Per-token targets, and refining a predicted P arrival into a pick.

**Targets.** Token j ends on sample `stride*j + stride - 1`. It is positive
once that sample is at or after P, and stays positive to the end of the trace.
So p_t means "an event is under way", and `dt` (seconds since P) says how long
ago it started. Together they separate a new onset (p high, dt small) from the
coda of an old one (p high, dt large). One probability alone cannot do that.

There is no label smoothing. Smoothing to 0.1/0.9 is what made the previous
detector's outputs pile up on a 0.90–0.907 plateau, where whether an Mw 6.2
triggered came down to the third decimal.

**Label uncertainty is data, not code.** Each trace carries
`p_tolerance_s`: tokens closer to P than that get weight 0, because their
label is a coin flip. It is 0.1 s for a manual pick, 0.3 s for an AIC-refined
one and 1.0 s for a bare TauP prediction (see docs/DATA.md).
"""
from __future__ import annotations

import numpy as np


def token_targets(n_tokens: int, stride: int, p_sample: float | None,
                  tolerance_samples: float, sample_rate: float, max_dt_s: float,
                  early_samples: float = 0, early_weight: float = 1.0,
                  pre_samples: float = 0, pre_weight: float = 1.0,
                  second_sample: float | None = None,
                  second_tolerance_samples: float = 0,
                  second_dt_samples: float = 0, second_dt_weight: float = 1.0) -> dict:
    """Targets for one trace.

    `second_sample` is the P of a second event inside the first one's coda
    (data.py, `_second`): `y` stays 1 through it, and `dt` restarts there,
    since that is how the trigger tells a new onset from the old coda. Tokens
    within `second_tolerance_samples` of it get no dt target, and the dt loss
    of the `second_dt_samples` after it is weighted `second_dt_weight`: the
    restart is what the trigger fires on, and it is a few tokens against a
    coda of many.

    Returns:
        dict of (n_tokens,) float32 arrays: `y` (0/1), `w` (loss weight),
        `dt` (seconds since the latest P, clipped), `dt_mask` (1 where `dt`
        is trained) and `dt_w` (the dt loss weight where it is).
    """
    ends = np.arange(n_tokens) * stride + stride - 1
    y = np.zeros(n_tokens, np.float32)
    w = np.ones(n_tokens, np.float32)
    dt = np.zeros(n_tokens, np.float32)
    dt_mask = np.zeros(n_tokens, np.float32)
    dt_w = np.ones(n_tokens, np.float32)
    if p_sample is None or not np.isfinite(p_sample):
        return {"y": y, "w": w, "dt": dt, "dt_mask": dt_mask, "dt_w": dt_w}

    rel = ends - p_sample
    after = rel >= 0
    y[after] = 1.0
    w[after & (rel < early_samples)] = early_weight
    w[~after & (rel >= -pre_samples)] = pre_weight
    unsure = np.abs(rel) < tolerance_samples
    w[unsure] = 0.0
    dt[after] = np.minimum(rel[after] / sample_rate, max_dt_s)
    dt_mask[after & ~unsure] = 1.0
    if second_sample is not None and np.isfinite(second_sample):
        rel2 = ends - second_sample
        after2 = rel2 >= 0
        dt[after2] = np.minimum(rel2[after2] / sample_rate, max_dt_s)
        dt_mask[np.abs(rel2) < second_tolerance_samples] = 0.0
        dt_w[after2 & (rel2 < second_tolerance_samples + second_dt_samples)] = second_dt_weight
    return {"y": y, "w": w, "dt": dt, "dt_mask": dt_mask, "dt_w": dt_w}


def aic_onset(x: np.ndarray) -> int:
    """Maeda's AIC change point: the index that best splits `x` into two
    stationary pieces of different variance."""
    n = len(x)
    if n < 8:
        raise ValueError("segment too short for an AIC pick")
    x = x.astype(np.float64) - x.mean()
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    k = np.arange(2, n - 2)
    var_a = c2[k - 1] / k - (c1[k - 1] / k) ** 2
    m = n - k
    var_b = (c2[-1] - c2[k - 1]) / m - ((c1[-1] - c1[k - 1]) / m) ** 2
    eps = 1e-12 * max(c2[-1] / n, 1e-30)
    aic = k * np.log(var_a + eps) + (m - 1) * np.log(var_b + eps)
    return int(k[np.argmin(aic)])


def refine_p(z: np.ndarray, missing: np.ndarray, p_guess: float, sample_rate: float,
             s_guess: float | None = None, before_s: float = 2.0, after_s: float = 3.0,
             min_snr: float = 2.0, settle_s: float = 1.5) -> tuple[float, bool, float]:
    """Moves a predicted P onto the waveform's onset, if it can find one.

    Searches `[p_guess - before_s, p_guess + after_s]` on the vertical,
    stopping 0.5 s short of the predicted S so a close station's S cannot win,
    and starting at least `settle_s` into the trace so the filter's start-up
    transient cannot win either. The window leans late because TauP's iasp91
    runs early here: on the KO pulls, accepted picks land a median ~1 s after
    the prediction.
    The pick is accepted only if the signal after it is `min_snr` times the
    noise before it (RMS over one second each side) and the window has no gap.

    Returns:
        (pick sample, accepted, snr). On rejection the pick is `p_guess`.
    """
    lo = int(max(settle_s * sample_rate, p_guess - before_s * sample_rate))
    hi = int(min(len(z), p_guess + after_s * sample_rate))
    if s_guess is not None and np.isfinite(s_guess):
        hi = min(hi, int(s_guess - 0.5 * sample_rate))
    if hi - lo < int(1.5 * sample_rate) or missing[lo:hi].any():
        return float(p_guess), False, float("nan")
    pick = lo + aic_onset(z[lo:hi])
    w = int(sample_rate)
    pre, post = z[max(0, pick - w): pick], z[pick: pick + w]
    if len(pre) < w // 2 or len(post) < w // 2:
        return float(p_guess), False, float("nan")
    snr = float(np.sqrt(np.mean(post ** 2) / max(np.mean(pre ** 2), 1e-30)))
    if snr < min_snr:
        return float(p_guess), False, snr
    return float(pick), True, snr
