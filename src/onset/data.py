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

**A second event in the coda** (`_second`). Every trace holds one event, so
without help the model never sees an onset inside another event's coda, and
it learns to read an aftershock there as more coda: on the Marmara M6.2
sequence the v2 model's dt did not restart for 116 of 296 arrivals. So
`second_p` of the training event crops get a second event trace added
`second_min_s`-`second_max_s` after the first P, from the same station where
it has one, scaled so its first 2 s are 1-`second_snr_max` times the RMS
just before it, and faded in over half a second a second ahead of its P. p
stays 1 through it and dt restarts at its P (`labels.token_targets`), with
the dt loss of the `second_dt_s` after it weighted `second_dt_weight`.
Evaluation gives every `eval_second_every`-th event trace one, at a fixed
draw, so validation measures the restart that ayzek's trigger fires on.
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
from onset.later import decode
from onset.store import StoreReader


MAX_LATER = 4                  # picked later onsets carried per example


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
        # Second-event sources: the split's event traces, by station; not ones
        # with a catalogued onset of their own after P, which would come along
        # unlabelled.
        ev = idx[(idx.split == split) & (idx.kind == "event") & idx.p_sample.notna()]
        if "later_p" in ev:
            ev = ev[ev.later_p.isna()]
        self.events = ev.reset_index(drop=True)
        self.events_by_station = {k: g.index.tolist()
                                  for k, g in self.events.groupby(["network", "station"])}
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

    def _second(self, r, wave, missing, p, rng):
        """Adds a second event trace into the coda after `p`, in place.

        Returns (its P sample, its P tolerance in samples), or (None, 0) when
        there is no room for one or no source trace.
        """
        fs, d = self.fs, self.data
        lo = p + d.second_min_s * fs
        hi = min(p + d.second_max_s * fs, len(wave) - 2 * fs)
        if hi <= lo or not len(self.events):
            return None, 0.0
        same = [j for j in self.events_by_station.get((r.network, r.station), [])
                if self.events.key[j] != r.key]
        j = (same[int(rng.integers(0, len(same)))] if same
             else int(rng.integers(0, len(self.events))))
        r2 = self.events.iloc[j]
        if r2.key == r.key:
            return None, 0.0
        w2, m2 = self.store.read(r2.key)
        p_src = int(r2.p_sample)
        lead, fade = int(1.0 * fs), int(0.5 * fs)
        if p_src < lead or p_src + 2 * fs > len(w2):
            return None, 0.0
        p2 = int(rng.uniform(lo, hi))
        n = min(len(w2) - (p_src - lead), len(wave) - (p2 - lead))
        seg = w2[p_src - lead: p_src - lead + n].astype(np.float32)
        seg_m = m2[p_src - lead: p_src - lead + n]
        ramp = np.ones(n, np.float32)
        ramp[:fade] = np.sin(np.linspace(0.0, np.pi / 2, fade, dtype=np.float32)) ** 2
        if n < len(wave) - (p2 - lead):                               # it ends early: fade out
            ramp[-fade:] = np.minimum(ramp[-fade:], ramp[:fade][::-1])
        before = wave[p2 - 2 * int(fs): p2][~missing[p2 - 2 * int(fs): p2]]
        first = seg[lead: lead + 2 * int(fs)][~seg_m[lead: lead + 2 * int(fs)]]
        if not len(before) or not len(first):
            return None, 0.0
        a, b = np.sqrt(np.mean(before ** 2)), np.sqrt(np.mean(first ** 2))
        if not (a > 0 and b > 0):
            return None, 0.0
        snr = np.exp(rng.uniform(0.0, np.log(d.second_snr_max)))
        s = p2 - lead
        wave[s: s + n] += (snr * a / b) * seg * ramp[:, None]
        missing[s: s + n] |= seg_m
        tol = float(r2.p_tolerance_s) if pd.notna(r2.p_tolerance_s) else 0.0
        return float(p2), tol * fs

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
        later = decode(r.get("later_p")) if p is not None else []
        p_stored = p
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
        wave, missing = wave[a:b].copy(), missing[a:b].copy()
        p_local = None if p is None else p - a
        # The trace's own later onsets, moved like P by the splice and the crop.
        later = [(s + (p - p_stored) - a, tol * self.fs, k == "a") for s, tol, k in later
                 if 0 <= s + (p - p_stored) - a < b - a] if p is not None else []
        p2, tol2 = None, 0.0
        if p_local is not None:
            if self.train and self.rng.random() < self.data.second_p:
                p2, tol2 = self._second(r, wave, missing, p_local, self.rng)
            elif (not self.train and self.data.eval_second_every
                  and i % self.data.eval_second_every == self.data.eval_second_every - 1):
                p2, tol2 = self._second(r, wave, missing, p_local,
                                        np.random.default_rng(10_000_019 + i))

        if self.train and ctx is not None and self.rng.random() < self.data.ctx_drop:
            ctx = None
        if self.train and self.rng.random() < self.data.gap_aug_p:
            protect = None if p_local is None else (int(p_local - self.fs), int(p_local + self.fs))
            if p2 is not None and self.rng.random() < 0.5:
                protect = (int(p2 - self.fs), int(p2 + self.fs))
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
                          self.data.pre_s * self.fs, self.data.pre_weight,
                          p2, tol2, self.data.second_dt_s * self.fs,
                          self.data.second_dt_weight, later,
                          self.data.later_mask_before_s * self.fs,
                          self.model.max_dt_s * self.fs)
        # Geometry is the first event's; after the next onset it is not.
        t["geo_mask"] = t["dt_mask"].copy()
        nxt = [p2 - tol2] if p2 is not None else []
        nxt += [s - tol - self.data.later_mask_before_s * self.fs for s, tol, _ in later]
        if nxt:
            ends = np.arange(len(t["y"])) * self.stride + self.stride - 1
            t["geo_mask"][ends >= min(nxt)] = 0.0
        return {"x": torch.from_numpy(x), "ctx": torch.from_numpy(c),
                "has_ctx": torch.tensor(has),
                **{k: torch.from_numpy(v) for k, v in t.items()},
                "is_event": torch.tensor(p is not None),
                "p_s": torch.tensor(np.nan if p_local is None else p_local / self.fs),
                "tol_s": torch.tensor(tol), "index": torch.tensor(i),
                # Geometry target; NaN where unknown (noise). The S time is
                # local, like P, for the evaluation.
                "dist_km": torch.tensor(float(r.distance_km) if p is not None
                                        and pd.notna(r.get("distance_km")) else np.nan),
                "s_s": torch.tensor(self._s_local(r, p, p_local)),
                "p2_s": torch.tensor(np.nan if p2 is None else p2 / self.fs),
                # Picked later onsets, for evaluation: local seconds, NaN-padded.
                "later_s": torch.tensor((sorted(s / self.fs for s, _, ok in later if ok)
                                         + [np.nan] * MAX_LATER)[:MAX_LATER]),
                "tol2_s": torch.tensor(tol2 / self.fs),
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
        elif k in ("y", "w", "dt", "dt_mask", "dt_w", "geo_mask"):
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
