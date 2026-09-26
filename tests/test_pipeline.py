"""Labels, filtering, metrics and the dataset, on synthetic data."""
import numpy as np
import pytest
import torch

from onset import metrics
from onset.conditioning import channel_scale, condition
from onset.config import DataConfig, ModelConfig
from onset.labels import aic_onset, refine_p, token_targets
from onset.store import StoreWriter, station_split

FS = 100.0


# -- labels -----------------------------------------------------------------

def test_targets_switch_on_at_the_token_that_contains_p():
    t = token_targets(10, 10, p_sample=35, tolerance_samples=0, sample_rate=FS, max_dt_s=10)
    # token 3 spans samples 30..39 and ends after P=35
    assert t["y"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    assert t["dt"][3] == pytest.approx(0.04)


def test_tolerance_band_is_ignored_not_labelled():
    t = token_targets(20, 10, p_sample=100, tolerance_samples=25, sample_rate=FS, max_dt_s=10)
    ends = np.arange(20) * 10 + 9
    assert (t["w"][np.abs(ends - 100) < 25] == 0).all()
    assert (t["w"][np.abs(ends - 100) >= 25] > 0).all()
    assert (t["dt_mask"][np.abs(ends - 100) < 25] == 0).all()


def test_noise_trace_is_all_negative():
    t = token_targets(8, 10, None, 0, FS, 10)
    assert t["y"].sum() == 0 and (t["w"] == 1).all()


def test_aic_finds_a_variance_step():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 400), rng.normal(0, 8, 300)])
    assert abs(aic_onset(x) - 400) <= 3


def test_refine_p_moves_a_bad_guess_onto_the_onset():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, 3000)
    z[1200:] += rng.normal(0, 10, 1800)
    pick, ok, snr = refine_p(z, np.zeros(3000, bool), p_guess=1100, sample_rate=FS)
    assert ok and abs(pick - 1200) <= 5 and snr > 5


def test_refine_p_rejects_pure_noise():
    z = np.random.default_rng(2).normal(0, 1, 3000)
    _, ok, _ = refine_p(z, np.zeros(3000, bool), p_guess=1500, sample_rate=FS)
    assert not ok


# -- filtering ----------------------------------------------------------------

def test_filter_is_causal_and_restarts_after_a_gap():
    pytest.importorskip("scipy")
    from onset.dsp import causal_filter
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 2000)
    missing = np.zeros(2000, bool)
    missing[1000:1050] = True
    y, m = causal_filter(x, missing)
    x2 = x.copy()
    x2[1500:] += 100.0
    y2, _ = causal_filter(x2, missing)
    assert np.allclose(y[:1500], y2[:1500])            # the future never reaches the past
    assert (y[1000:1050] == 0).all() and m[1000:1050].all()
    x3 = x.copy()
    x3[:1000] *= 50                                     # the run before a gap ...
    y3, _ = causal_filter(x3, missing)
    assert np.allclose(y[1050:], y3[1050:])            # ... cannot leak past it


def test_short_runs_become_missing():
    pytest.importorskip("scipy")
    from onset.dsp import causal_filter
    missing = np.ones(1000, bool)
    missing[400:450] = False                           # 0.5 s of data
    _, m = causal_filter(np.ones(1000), missing)
    assert m.all()


# -- conditioning -------------------------------------------------------------

def test_conditioning_zeroes_gaps_and_flags_them():
    x = np.ones((20, 3), np.float32) * 3
    missing = np.zeros(20, bool)
    missing[5:8] = True
    out = condition(x, missing, np.array([3.0, 3.0, 3.0], np.float32))
    assert np.allclose(out[0, :3], np.arcsinh(1.0))
    assert (out[5:8, :3] == 0).all() and (out[5:8, 3] == 1).all() and out[0, 3] == 0


def test_scale_ignores_missing_samples_and_dead_channels():
    x = np.zeros((10, 3), np.float32)
    x[:, 0] = 2.0
    x[:5, 1] = 1000.0
    missing = np.zeros(10, bool)
    missing[:5] = True
    s = channel_scale(x, missing)
    assert s[0] == pytest.approx(2.0) and s[1] == 1.0 and s[2] == 1.0


# -- metrics -----------------------------------------------------------------

def test_rising_edges_have_hysteresis():
    p = np.array([0.1, 0.9, 0.6, 0.95, 0.3, 0.2, 0.9, 0.1])
    assert metrics.rising_edges(p, 0.8, 0.4).tolist() == [1, 6]


def test_event_latency_and_early_trigger():
    stride, fs = 10, FS
    p = np.zeros(100)
    p[52:] = 0.99                         # token 52 ends at 5.29 s
    lat, early, _ = metrics.score_event(p, np.zeros(100), 5.0, 0.3, stride, fs, 0.9, 0.45)
    assert lat == pytest.approx(0.29) and not early
    p[10] = 0.99                          # a pre-P blip
    lat, early, _ = metrics.score_event(p, np.zeros(100), 5.0, 0.3, stride, fs, 0.9, 0.45)
    assert early and lat == pytest.approx(0.29)


def test_sweep_counts_false_triggers_per_hour():
    noise = [{"p": np.tile([0.0, 0.99], 1800), "missing_tokens": np.zeros(3600, bool)}]
    rows = metrics.sweep([], noise, 10, FS, thresholds=[0.9])
    assert rows[0]["false_per_hour"] == pytest.approx(1800 / 0.1)


# -- dataset -----------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(4)
    w = StoreWriter(tmp_path)
    sta = next(s for s in (f"S{i}" for i in range(100)) if station_split("XX", s) == "train")
    base = {"split": "train", "source": "fdsn", "network": "XX", "station": sta}
    ctx = rng.normal(0, 1, (12000, 3)).astype(np.float32)
    w.add("context/1/XX.S", ctx, np.zeros(12000, bool), {**base, "kind": "context"})
    ev = rng.normal(0, 1, (6000, 3)).astype(np.float32)
    ev[700:] *= 20
    w.add("event/1/XX.S", ev, np.zeros(6000, bool),
          {**base, "kind": "event", "p_sample": 700.0, "p_source": "aic",
           "p_tolerance_s": 0.3, "context_key": "context/1/XX.S", "magnitude": 3.0})
    w.add("noise/1/XX.S", rng.normal(0, 1, (12000, 3)).astype(np.float32),
          np.zeros(12000, bool), {**base, "kind": "noise", "context_key": "context/1/XX.S"})
    w.close()
    return tmp_path


def test_training_crop_keeps_p_and_a_second_of_noise(store):
    from onset.data import OnsetDataset
    ds = OnsetDataset(store, "train", DataConfig(seq_seconds=20, gap_aug_p=0), ModelConfig(), True)
    ev = int(np.flatnonzero(ds.rows.kind == "event")[0])
    for seed in range(20):
        ds.reseed(seed)
        b = ds[ev]
        assert b["x"].shape == (2000, 4) and b["y"].shape == (200,)
        assert 1.0 - 1e-6 <= float(b["p_s"]) <= 19.0
        assert b["y"].sum() > 0 and b["y"][0] == 0


def test_evaluation_uses_whole_traces_and_context(store):
    from onset.data import OnsetDataset
    ds = OnsetDataset(store, "train", DataConfig(eval_lead_in_s=0), ModelConfig(), False)
    kinds = ds.rows.kind.tolist()
    ev, no = ds[kinds.index("event")], ds[kinds.index("noise")]
    assert ev["x"].shape == (6000, 4) and bool(ev["has_ctx"])
    assert float(ev["p_s"]) == pytest.approx(7.0)
    assert no["x"].shape == (12000, 4) and no["y"].sum() == 0
    assert ev["ctx"].shape == (6000, 4)


def test_a_batch_trains(store):
    from torch.utils.data import DataLoader
    from onset.data import OnsetDataset
    from onset.model import OnsetDetector
    from onset.train import loss_fn
    mc = ModelConfig(d_model=32, n_layers=2, window_tokens=20)
    ds = OnsetDataset(store, "train", DataConfig(seq_seconds=20), mc, True)
    b = next(iter(DataLoader(ds, batch_size=2)))
    out = OnsetDetector(mc)(b["x"], b["ctx"], b["has_ctx"])
    loss, _, _ = loss_fn(out, b, 0.1)
    loss.backward()
    assert torch.isfinite(loss)


def test_lead_in_spreads_the_history_before_p(store):
    """Without the lead-in an FDSN onset never has more than ~15 s of history;
    with it, the crop can put P late in the sequence."""
    from onset.data import OnsetDataset
    ds = OnsetDataset(store, "train", DataConfig(seq_seconds=40, gap_aug_p=0, lead_in_p=1.0,
                                                 ctx_drop=0), ModelConfig(), True)
    ev = int(np.flatnonzero(ds.rows.kind == "event")[0])
    seen = []
    for seed in range(60):
        ds.reseed(seed)
        b = ds[ev]
        seen.append(float(b["p_s"]))
        assert b["x"].shape == (4000, 4) and b["y"].sum() > 0
    assert max(seen) > 20.0 and min(seen) >= 1.0 - 1e-6


def test_the_splice_leaves_no_seam(store):
    """The lead-in joins without a gap and at the event's own noise level: a
    gap or a level step would be a cue that onsets follow joins."""
    from onset.data import OnsetDataset
    ds = OnsetDataset(store, "train", DataConfig(), ModelConfig(), False)
    r = ds.rows[ds.rows.kind == "event"].iloc[0]
    wave, missing = ds.store.read(r.key)
    w, m, p = ds._splice(r, wave, missing, float(r.p_sample), np.random.default_rng(0), 30.0)
    assert not m.any()
    assert p == pytest.approx(700 - 150 + 3000)          # trimmed 1.5 s, added 30 s
    seam = 3000
    before = np.sqrt((w[seam - 300: seam - 50] ** 2).mean(axis=0))
    after = np.sqrt((w[seam + 50: seam + 300] ** 2).mean(axis=0))
    assert np.allclose(before, after, rtol=0.35)


def test_evaluation_splices_past_the_lookback(store):
    from onset.data import OnsetDataset, pad_collate
    mc = ModelConfig()
    ds = OnsetDataset(store, "train", DataConfig(eval_lead_in_s=40), mc, False)
    kinds = ds.rows.kind.tolist()
    ev = ds[kinds.index("event")]
    assert float(ev["p_s"]) == pytest.approx(7.0 - 1.5 + 40.0)
    assert float(ev["p_s"]) > mc.lookback_tokens * mc.token_seconds
    no = ds[kinds.index("noise")]
    b = pad_collate([ev, no])
    assert b["x"].shape[1] == max(len(ev["x"]), len(no["x"]))
    short = min(ev, no, key=lambda it: len(it["x"]))
    k = [ev, no].index(short)
    assert (b["x"][k, len(short["x"]):, 3] == 1).all()
    assert int(b["n_tokens"][k]) == len(short["y"])


# -- geometry head -----------------------------------------------------------

def test_geometry_head_trains_and_leaves_detection_alone(store):
    """The head adds outputs; with it off, the model is the detector as before."""
    from torch.utils.data import DataLoader
    from onset.data import OnsetDataset
    from onset.model import OnsetDetector
    from onset.train import loss_fn
    mc = ModelConfig(d_model=32, n_layers=2, window_tokens=20, geometry=1)
    ds = OnsetDataset(store, "train", DataConfig(seq_seconds=20), mc, True)
    b = next(iter(DataLoader(ds, batch_size=3)))
    out = OnsetDetector(mc)(b["x"], b["ctx"], b["has_ctx"])
    assert {"log_dist", "log_dist_var", "baz_vec", "baz_log_kappa"} <= set(out)
    loss, _, _ = loss_fn(out, b, 0.1, geo_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert "log_dist" not in OnsetDetector(ModelConfig(d_model=32, n_layers=2))(b["x"])


def test_geometry_table_reads_errors_by_time_since_p():
    t_tok = 300
    ev = {"p": np.zeros(t_tok), "missing_tokens": np.zeros(t_tok, bool), "p_s": 5.0,
          "s_s": 12.0, "dist_km": 40.0, "baz_rad": np.radians(90.0),
          "log_dist": np.full(t_tok, np.log(44.0)), "log_dist_var": np.full(t_tok, np.log(0.2 ** 2)),
          "baz": np.full(t_tok, np.radians(100.0))}
    rows = {r["window"]: r for r in metrics.geometry_table([ev], 10, FS)}
    r = rows["1-2 s after P"]
    assert r["dist_abs_err_km_p50"] == pytest.approx(4.0, rel=1e-6)
    assert r["baz_err_deg_p50"] == pytest.approx(10.0, abs=1e-6)
    assert r["cal_1sd"] == 1.0                     # |log(44/40)| = 0.095 < 0.2
    assert "after S" in rows and "after P, before S" in rows
