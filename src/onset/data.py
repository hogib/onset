"""Stored traces -> training and evaluation examples.

One example is a waveform sequence, an optional station context, and a target
for every 0.1 s token (see `labels.token_targets`).

Training crops a `seq_seconds` window. For an event trace the crop starts
anywhere between the trace start and one second before P, so every crop holds
P and at least a second of pre-P noise, and the model trains with varying
amounts of history before the onset. Because the model is causal, where P
falls *within* the crop does not matter: the token half a second after P sees
the same past whether the crop ends one second later or thirty.

**Lead-in: why every trace is spliced onto older noise.** An FDSN event
trace starts at origin time, so its P always comes 2–11 s after the data
begins. A model trained on that learns "onsets come shortly after the data
starts". On a continuous stream the data never starts, and the model then
waits for the S wave: measured on the same 8 events, a model trained that way
fired 3–8 s after origin on the event trace alone and 7–24 s after origin on
the same trace behind 24 h of continuous data.

A first attempt joined older noise in front *across a gap*. That moved the
cue without removing it: the onset now came 2–11 s after the gap instead.
The model was just as blind on continuous data, and the validation set,
whose traces also started at origin, could not show it.

So the join is now seamless (`_splice`). The event trace loses its first 1.5 s
(the filter's start-up); the lead-in, from the same station's noise, is scaled
per component to the RMS of the event's own pre-P noise; and the two are
joined with a 0.5 s equal-power crossfade. There is no gap, no level step, and
no change in the noise's character to anchor on. Noise traces get the same
splice at the same rate, so even a seam the model could find says nothing
about the label.

**Evaluation** takes whole traces with a fixed `eval_lead_in_s` spliced in
front, longer than the model's lookback, so validation, too, measures the
detector with no view of where the data begins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from onset.conditioning import channel_scale, condition
from onset.config import DataConfig, ModelConfig
from onset.labels import token_targets
from onset.store import StoreReader


class OnsetDataset(Dataset):
    def __init__(self, root, split: str, data: DataConfig, model: ModelConfig,
                 train: bool, kinds=("event", "noise")):
        self.store = StoreReader(root)
        self.root = Path(root)
        idx = self.store.index
        self.rows = idx[(idx.split == split) & idx.kind.isin(kinds)].reset_index(drop=True)
        self.contexts = set(idx.loc[idx.kind == "context", "key"])
        # Lead-in sources: every noise-like trace of each station. Stations are
        # split-disjoint, so these never cross a split.
        pool = idx[idx.kind.isin(("noise", "context"))]
        self.pool = {k: g["key"].tolist() for k, g in pool.groupby(["network", "station"])}
        self.data, self.model, self.train = data, model, train
        self.fs = model.sample_rate
        self.stride = model.stride
        self.ctx_len = int(round(data.ctx_seconds * self.fs / self.stride)) * self.stride
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.rows)

    def reseed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    # -- pieces -----------------------------------------------------------

    def _crop(self, n: int, p: float | None) -> tuple[int, int]:
        s = self.stride
        if not self.train:
            return 0, (n // s) * s
        L = min(int(self.data.seq_seconds * self.fs) // s * s, (n // s) * s)
        if p is None:
            a = int(self.rng.integers(0, n - L + 1))
        else:
            hi = int(min(n - L, max(0, p - self.data.fallback_scale_s * self.fs)))
            lo = int(max(0, min(hi, p - L + self.fs)))      # P at least 1 s before the end
            a = int(self.rng.integers(lo, hi + 1))
        return a, a + L

    def _context(self, key) -> tuple[np.ndarray, np.ndarray] | None:
        if not isinstance(key, str) or key not in self.contexts:
            return None
        wave, missing = self.store.read(key)
        L = self.ctx_len
        if len(wave) < L:
            return None
        a = int(self.rng.integers(0, len(wave) - L + 1)) if self.train else len(wave) - L
        wave, missing = wave[a:a + L], missing[a:a + L]
        return (wave, missing) if missing.mean() < 0.5 else None

    def _lead_source(self, r, n: int, rng):
        """`n` samples of the station's own noise, nearly gap-free, or None.
        The event's context when it has one, else any noise-like trace of the
        station."""
        keys = [r.context_key] if isinstance(r.context_key, str) else []
        keys += [k for k in self.pool.get((r.network, r.station), []) if k != r.key]
        if not keys:
            return None
        key = keys[0] if not self.train else keys[int(rng.integers(0, len(keys)))]
        w, m = self.store.read(key)
        if len(w) < n:
            return None
        a = int(rng.integers(0, len(w) - n + 1)) if self.train else len(w) - n
        w, m = w[a:a + n], m[a:a + n]
        return (w, m) if m.mean() < 0.05 else None

    def _splice(self, r, wave, missing, p, rng, seconds: float):
        """Joins `seconds` of the station's older noise seamlessly in front.

        Returns (wave, missing, p) with P moved, or None when the trace cannot
        take a splice: no noise for the station, or P too close to the start
        to trim the filter start-up and still crossfade before it.
        """
        fs = self.fs
        trim = int(self.data.splice_trim_s * fs)
        xf = int(self.data.splice_xfade_s * fs)
        if p is not None:
            ref_end = int(p - 0.5 * fs)           # pre-P noise, clear of the onset
            if ref_end - trim < xf + int(0.5 * fs):
                return None
        else:
            ref_end = trim + int(5 * fs)
        n = int(seconds * fs) + xf
        src = self._lead_source(r, n, rng)
        if src is None:
            return None
        lead, lead_m = src
        body, body_m = wave[trim:], missing[trim:]
        ref = channel_scale(body[: ref_end - trim], body_m[: ref_end - trim])
        lead = lead * (ref / channel_scale(lead, lead_m))
        th = np.linspace(0.0, np.pi / 2, xf, dtype=np.float32)[:, None]
        seam = lead[-xf:] * np.cos(th) + body[:xf] * np.sin(th)
        w = np.concatenate([lead[:-xf], seam, body[xf:]]).astype(np.float32)
        m = np.concatenate([lead_m[:-xf], lead_m[-xf:] | body_m[:xf], body_m[xf:]])
        shift = len(lead) - xf - trim
        return w, m, None if p is None else p + shift

    def _s_local(self, r, p, p_local):
        """The predicted S in crop-local seconds: S keeps its offset from P
        through splices and crops."""
        if p is None or pd.isna(r.get("s_sample")) or pd.isna(r.get("p_sample")):
            return np.nan
        return (p_local + float(r.s_sample) - float(r.p_sample)) / self.fs

    def _gap(self, missing: np.ndarray, protect: tuple[int, int] | None):
        """A synthetic gap, as ayzek sees them on the horizontals every day.
        Never over the labelled onset itself: a gap there leaves no signal to
        train on, only a label."""
        L = len(missing)
        g = int(self.rng.uniform(0.2, self.data.gap_aug_max_s) * self.fs)
        if g >= L:
            return
        a = int(self.rng.integers(0, L - g))
        if protect is not None and a < protect[1] and a + g > protect[0]:
            return
        missing[a:a + g] = True

    # -- item -------------------------------------------------------------

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        wave, missing = self.store.read(r.key)
        p = float(r.p_sample) if r.kind == "event" and pd.notna(r.p_sample) else None
        ctx = self._context(r.context_key)
        if self.train:
            if self.rng.random() < self.data.lead_in_p:
                secs = self.rng.uniform(5.0, self.data.lead_in_max_s)
                got = self._splice(r, wave, missing, p, self.rng, secs)
                if got is not None:
                    wave, missing, p = got
        elif self.data.eval_lead_in_s > 0:
            got = self._splice(r, wave, missing, p, np.random.default_rng(i),
                               self.data.eval_lead_in_s)
            if got is not None:
                wave, missing, p = got
        a, b = self._crop(len(wave), p)
        wave, missing = wave[a:b], missing[a:b].copy()
        p_local = None if p is None else p - a

        if self.train and ctx is not None and self.rng.random() < self.data.ctx_drop:
            ctx = None
        if self.train and self.rng.random() < self.data.gap_aug_p:
            protect = None if p_local is None else (int(p_local - self.fs), int(p_local + self.fs))
            self._gap(missing, protect)

        if ctx is not None:
            scale = channel_scale(*ctx)
        else:
            n0 = int(self.data.fallback_scale_s * self.fs)
            scale = channel_scale(wave[:n0], missing[:n0])
        x = condition(wave, missing, scale)
        if ctx is not None:
            c, has = condition(ctx[0], ctx[1], scale), True
        else:
            c, has = np.zeros((self.ctx_len, 4), np.float32), False

        tol = float(r.p_tolerance_s) if pd.notna(r.p_tolerance_s) else 0.0
        t = token_targets(len(x) // self.stride, self.stride, p_local, tol * self.fs,
                          self.fs, self.model.max_dt_s,
                          self.data.early_s * self.fs, self.data.early_weight,
                          self.data.pre_s * self.fs, self.data.pre_weight)
        return {"x": torch.from_numpy(x), "ctx": torch.from_numpy(c),
                "has_ctx": torch.tensor(has),
                **{k: torch.from_numpy(v) for k, v in t.items()},
                "is_event": torch.tensor(p is not None),
                "p_s": torch.tensor(np.nan if p_local is None else p_local / self.fs),
                "tol_s": torch.tensor(tol), "index": torch.tensor(i),
                # Geometry targets; NaN where unknown (noise, or a Z12 instrument's
                # back-azimuth). The S time is local, like P, for the evaluation.
                "dist_km": torch.tensor(float(r.distance_km) if p is not None
                                        and pd.notna(r.get("distance_km")) else np.nan),
                "baz_rad": torch.tensor(np.radians(float(r.back_azimuth_deg))
                                        if p is not None and pd.notna(r.get("back_azimuth_deg"))
                                        else np.nan),
                "s_s": torch.tensor(self._s_local(r, p, p_local)),
                "n_tokens": torch.tensor(len(x) // self.stride)}


def pad_collate(items):
    """Batches examples of different lengths (evaluation, where a splice
    lengthens some traces and not others). Padding goes at the end, marked
    missing; the model is causal, so it cannot change any real token, and
    `n_tokens` says where each example's real tokens stop."""
    L = max(len(it["x"]) for it in items)
    stride = len(items[0]["x"]) // len(items[0]["y"])
    out = {}
    for k in items[0]:
        vals = [it[k] for it in items]
        if k == "x":
            n = L
        elif k in ("y", "w", "dt", "dt_mask"):
            n = L // stride
        else:
            out[k] = torch.stack(vals)          # scalars, and the fixed-length context
            continue
        padded = []
        for v in vals:
            fill = torch.zeros((n - len(v),) + tuple(v.shape[1:]), dtype=v.dtype)
            if k == "x":
                fill[:, 3] = 1.0
            padded.append(torch.cat([v, fill]))
        out[k] = torch.stack(padded)
    return out
