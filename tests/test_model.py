"""The properties the detector's design rests on.

Causality and exact streaming are not implementation details here: a model
that peeks one sample ahead reports a latency it cannot achieve, and a
streaming path that drifts from the trained forward pass deploys a different
model from the one that was evaluated.
"""
import numpy as np
import pytest
import torch

from onset.config import ModelConfig
from onset.model import OnsetDetector
from onset.stream import StreamingDetector, score_blocks

SMALL = dict(d_model=32, n_heads=4, n_layers=2, window_tokens=12, context_tokens=4,
             dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return OnsetDetector(ModelConfig(**SMALL)).eval()


def run(model, x, ctx=None):
    has = None if ctx is None else torch.tensor([True])
    with torch.no_grad():
        out = model(x[None], None if ctx is None else ctx[None], has)
    return out["logit"][0], out["dt"][0]


def signal(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    x[:, 3] = 0
    return x


def test_one_token_per_stride(model):
    assert run(model, signal(400))[0].shape == (400 // model.cfg.stride,)


def test_rejects_partial_tokens(model):
    with pytest.raises(ValueError, match="multiple"):
        run(model, signal(405))


def test_future_samples_never_reach_past_tokens(model):
    x = signal(600)
    base = run(model, x)[0]
    y = x.clone()
    y[300:] += 5.0                          # token 30 starts at sample 300
    moved = run(model, y)[0]
    assert torch.allclose(base[:30], moved[:30], atol=1e-6)
    assert not torch.allclose(base[30:], moved[30:])


def test_a_token_sees_the_last_sample_of_its_own_span(model):
    """Token j must end on sample 10j+9, not lag it."""
    x = signal(400)
    base = run(model, x)[0]
    y = x.clone()
    y[209] += 5.0                           # last sample of token 20
    moved = run(model, y)[0]
    assert torch.allclose(base[:20], moved[:20], atol=1e-6)
    assert not torch.isclose(base[20], moved[20])


def test_lookback_is_bounded(model):
    """Samples older than the stacked windows plus the stem cannot matter."""
    cfg = model.cfg
    x = signal(1200)
    base = run(model, x)[0]
    y = x.clone()
    y[:50] += 5.0
    moved = run(model, y)[0]
    first_free = cfg.lookback_tokens + 50 // cfg.stride + 2
    assert torch.allclose(base[first_free:], moved[first_free:], atol=1e-5)


@pytest.mark.parametrize("with_ctx", [False, True])
def test_streaming_matches_the_forward_pass(model, with_ctx):
    x = signal(700)
    ctx = signal(300, seed=7) if with_ctx else None
    logit, dt = run(model, x, ctx)
    s = StreamingDetector(model, ctx)
    got = []
    for chunk in torch.split(x, 37):         # arrival size unrelated to the stride
        got += s.push(chunk)
    p = torch.tensor([g[0] for g in got])
    d = torch.tensor([g[1] for g in got])
    assert len(got) == len(logit)
    assert torch.allclose(p, torch.sigmoid(logit), atol=1e-5)
    assert torch.allclose(d, dt, atol=1e-4)


def test_block_scoring_matches_the_forward_pass(model):
    x = signal(3000)
    logit, dt = run(model, x)
    probs, dts = score_blocks(model, x.numpy(), block_tokens=35)
    assert np.allclose(probs, torch.sigmoid(logit).numpy(), atol=1e-5)
    assert np.allclose(dts, dt.numpy(), atol=1e-4)


def test_context_changes_the_output_and_absent_context_uses_the_null(model):
    x = signal(400)
    none = run(model, x)[0]
    some = run(model, x, signal(300, seed=3))[0]
    assert not torch.allclose(none, some)
    with torch.no_grad():
        out = model(x[None], signal(300, seed=3)[None], torch.tensor([False]))
    assert torch.allclose(out["logit"][0], none, atol=1e-6)


def test_gapped_context_stays_finite(model):
    ctx = signal(300, seed=3)
    ctx[:, :3] = 0
    ctx[:, 3] = 1
    assert torch.isfinite(run(model, signal(400), ctx)[0]).all()


def test_streaming_geometry_matches_the_forward_pass():
    """The geometry head streams exactly too: ayzek locates from these outputs."""
    from onset.locate import estimates_from_head
    torch.manual_seed(0)
    m = OnsetDetector(ModelConfig(**SMALL, geometry=1)).eval()
    x = signal(400)
    with torch.no_grad():
        out = m(x[None])
    s = StreamingDetector(m)
    got = []
    for chunk in torch.split(x, 23):
        got += s.push(chunk, geometry=True)
    assert len(got) == out["log_dist"].shape[1]
    for t in (0, 17, len(got) - 1):
        ref = estimates_from_head(out, token=t)
        for k, v in ref.items():
            assert got[t][2][k] == pytest.approx(v, rel=1e-4, abs=1e-5), (t, k)
    with pytest.raises(ValueError):
        StreamingDetector(OnsetDetector(ModelConfig(**SMALL))).push(x, geometry=True)


def test_geometry_uncertainty_has_a_floor():
    torch.manual_seed(0)
    m = OnsetDetector(ModelConfig(**SMALL, geometry=1, geo_min_sd=0.1)).eval()
    with torch.no_grad():
        m.geo_head.bias[1] = -50.0                # the head trying to state sd ~ 0
        out = m(signal(200)[None])
    assert torch.allclose(out["log_dist_var"].min(), torch.tensor(2 * np.log(0.1), dtype=torch.float32), atol=1e-5)
