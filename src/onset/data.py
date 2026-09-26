"""Stored traces -> training and evaluation examples.

One example is a waveform sequence, an optional station context, and a target
for every 0.1 s token (see `labels.token_targets`).

Training crops a `seq_seconds` window. For an event trace the crop starts
anywhere between the trace start and one second before P, so every crop holds
P and at least a second of pre-P noise, and the model trains with varying
amounts of history before the onset. Because the model is causal, where P
falls *within* the crop does not matter: the token half a second after P sees
the same past whether the crop ends one second later or thirty.

**Lead-in.** An FDSN event trace starts at origin time, so its P has at most
~15 s of history, while noise crops have up to `seq_seconds`. Left alone, that
is a shortcut: through its ~32 s stacked lookback the model could learn that
onsets never come late in a sequence, and fail exactly on a continuous stream.
With probability `lead_in_p`, a trace (event or noise alike) is prefixed with
up to `lead_in_max_s` of the same station's context noise and a 0.2–1 s gap
between them. The crop then puts P anywhere from 1 s to the crop's end, and
because noise gets the same treatment, "a gap came before" says nothing about
the label. The join is a gap and not a splice because a gap is real: the
filter restarts there and the mask channel says so, as it would on a live
station.

Evaluation takes whole traces, uncropped.
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

    def _lead_in(self, wave, missing, ctx):
        """Prefix `wave` with context noise and a gap; returns the new arrays and
        how many samples were added in front."""
        lead_w, lead_m = ctx
        n = int(self.rng.uniform(1.0, self.data.lead_in_max_s) * self.fs)
        n = min(n, len(lead_w))
        g = int(self.rng.uniform(0.2, 1.0) * self.fs)
        w = np.concatenate([lead_w[-n:], np.zeros((g, 3), np.float32), wave])
        m = np.concatenate([lead_m[-n:], np.ones(g, bool), missing])
        return w, m, n + g

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
        if self.train and ctx is not None and self.rng.random() < self.data.lead_in_p:
            wave, missing, shift = self._lead_in(wave, missing, ctx)
            p = None if p is None else p + shift
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
                "tol_s": torch.tensor(tol), "index": torch.tensor(i)}
