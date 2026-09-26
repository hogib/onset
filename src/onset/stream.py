"""Running the detector on a stream.

Two ways, with different jobs:

- `StreamingDetector` is the deployment algorithm: samples in, one output per
  0.1 s token, a key/value cache per layer and a short raw-sample history for
  the stem. It is what a C++ port transcribes, and the tests hold its outputs
  equal to `OnsetDetector.forward` on the same samples.
- `score_blocks` produces the same outputs for a long recording in blocks of a
  full forward pass each, with enough overlap (`block_overlap_tokens`) that
  every output token sees its whole receptive field. `replay` uses the same
  overlap to score a day of data in seconds instead of the ~10 minutes
  one-token-at-a-time Python would take.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from onset.model import OnsetDetector


class StreamingDetector:
    """One station's detector state. Feed it conditioned samples with `push`."""

    def __init__(self, model: OnsetDetector, ctx: torch.Tensor | None = None):
        """
        Args:
            model: a trained detector; it is put in eval mode.
            ctx: (Tc, 4) conditioned station context, or None for the null
                context. Change it later with `set_context`.
        """
        self.model = model.eval()
        cfg = model.cfg
        self.stride = cfg.stride
        self.window = cfg.window_tokens
        # History the stem needs for the newest token, rounded to whole tokens
        # so a slice of it always starts on the token grid.
        self.history = math.ceil(cfg.stem_receptive_field / self.stride) * self.stride
        self.device = next(model.parameters()).device
        self.samples = torch.zeros(0, cfg.in_channels, device=self.device)
        self.pending = torch.zeros(0, cfg.in_channels, device=self.device)
        self.n_seen = 0
        self.cache = [None] * cfg.n_layers
        self.set_context(ctx)

    @torch.no_grad()
    def set_context(self, ctx: torch.Tensor | None):
        has = None if ctx is None else torch.tensor([True], device=self.device)
        c = None if ctx is None else ctx[None].to(self.device)
        self.ctx = self.model.encode_context(c, has, 1)

    @torch.no_grad()
    def push(self, x: torch.Tensor, geometry: bool = False) -> list[tuple]:
        """Adds (N, 4) samples; returns `(probability, dt)` for each token they
        complete, oldest first. With `geometry` (a model trained with the
        geometry head), `(probability, dt, estimate)`, the estimate as
        `locate.estimates_from_head` gives it."""
        if geometry and not self.model.cfg.geometry:
            raise ValueError("this model has no geometry head")
        self.pending = torch.cat([self.pending, x.to(self.device)])
        out = []
        while len(self.pending) >= self.stride:
            chunk, self.pending = self.pending[: self.stride], self.pending[self.stride:]
            out.append(self._token(chunk, geometry))
        return out

    def _token(self, chunk, geometry=False):
        # Until `history` samples exist the stem sees the stream from its first
        # sample, exactly as a full forward pass pads it; after that, only the
        # newest `history` samples can reach the newest token.
        self.samples = torch.cat([self.samples, chunk])[-self.history:]
        self.n_seen += self.stride
        e = self.model.stem(self.samples[None])[:, -1:]         # (1, 1, d)
        slopes = self.model.slopes
        for i, block in enumerate(self.model.blocks):
            h = block.ln_sa(e)
            k, v = block.sa.project_kv(h)
            if self.cache[i] is not None:
                k = torch.cat([self.cache[i][0], k], dim=2)[:, :, -self.window:]
                v = torch.cat([self.cache[i][1], v], dim=2)[:, :, -self.window:]
            self.cache[i] = (k, v)
            dist = torch.arange(k.shape[2] - 1, -1, -1, device=self.device).float()
            bias = (-slopes[:, None] * dist[None])[None, :, None, :]   # (1, H, 1, S)
            e = e + block.sa.attend(block.sa.split(block.sa.q(h)), k, v, bias)
            e = e + block.ca(block.ln_ca(e), self.ctx)
            e = e + block.ff(block.ln_ff(e))
        logit, dt = self.model.readout(e)
        if not geometry:
            return float(torch.sigmoid(logit)), float(dt)
        from onset.locate import estimates_from_head
        return (float(torch.sigmoid(logit)), float(dt),
                estimates_from_head(self.model.geometry(e), token=0))


def block_overlap_tokens(model: OnsetDetector) -> int:
    """Tokens of lead-in a block needs so its first output is exact: the
    stacked attention windows plus the stem's receptive field."""
    cfg = model.cfg
    return cfg.lookback_tokens - 1 + math.ceil(cfg.stem_receptive_field / cfg.stride)


@torch.no_grad()
def score_blocks(model: OnsetDetector, x: np.ndarray, ctx=None, block_tokens: int = 600):
    """Scores a long conditioned recording in overlapping blocks.

    Args:
        model: the detector.
        x: (T, 4) conditioned samples; T is truncated to whole tokens.
        ctx: (Tc, 4) conditioned context, or None.
        block_tokens: new output tokens per block.

    Returns:
        (probabilities, dt), each (T // stride,), equal to one forward pass
        over all of `x`.
    """
    model.eval()
    s = model.cfg.stride
    dev = next(model.parameters()).device
    n_tok = len(x) // s
    lead = block_overlap_tokens(model)
    c = None if ctx is None else torch.as_tensor(ctx, device=dev)[None]
    has = None if ctx is None else torch.tensor([True], device=dev)
    probs = np.zeros(n_tok, dtype=np.float32)
    dts = np.zeros(n_tok, dtype=np.float32)
    for t0 in range(0, n_tok, block_tokens):
        t1 = min(t0 + block_tokens, n_tok)
        a = max(0, t0 - lead)
        out = model(torch.as_tensor(x[a * s: t1 * s], device=dev)[None], c, has)
        probs[t0:t1] = torch.sigmoid(out["logit"][0, t0 - a:]).float().cpu().numpy()
        dts[t0:t1] = out["dt"][0, t0 - a:].float().cpu().numpy()
    return probs, dts
