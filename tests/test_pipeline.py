"""Labels, filtering, metrics and the dataset, on synthetic data."""
import numpy as np
import pandas as pd
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
    assert {"log_dist", "log_dist_var"} <= set(out)
    assert not {"baz_vec", "baz_log_kappa"} & set(out)
    loss, _, _ = loss_fn(out, b, 0.1, geo_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert "log_dist" not in OnsetDetector(ModelConfig(d_model=32, n_layers=2))(b["x"])


def test_geometry_table_reads_errors_by_time_since_p():
    t_tok = 300
    ev = {"p": np.zeros(t_tok), "missing_tokens": np.zeros(t_tok, bool), "p_s": 5.0,
          "s_s": 12.0, "dist_km": 40.0,
          "log_dist": np.full(t_tok, np.log(44.0)), "log_dist_var": np.full(t_tok, np.log(0.2 ** 2))}
    rows = {r["window"]: r for r in metrics.geometry_table([ev], 10, FS)}
    r = rows["1-2 s after P"]
    assert r["dist_abs_err_km_p50"] == pytest.approx(4.0, rel=1e-6)
    assert r["cal_1sd"] == 1.0                     # |log(44/40)| = 0.095 < 0.2
    assert "after S" in rows and "after P, before S" in rows


# -- second onsets and the deployed trigger ------------------------------------

def test_second_onset_restarts_dt_and_keeps_y():
    t = token_targets(100, 10, p_sample=205, tolerance_samples=0, sample_rate=FS, max_dt_s=10,
                      second_sample=605, second_tolerance_samples=15)
    ends = np.arange(100) * 10 + 9
    assert (t["y"][ends >= 205] == 1).all()
    assert t["dt"][ends == 599][0] == pytest.approx(3.94)          # the first event's
    assert t["dt"][ends == 619][0] == pytest.approx(0.14)          # restarted at 605
    assert (t["dt_mask"][np.abs(ends - 605) < 15] == 0).all()


def test_second_onset_restart_is_upweighted():
    t = token_targets(100, 10, p_sample=205, tolerance_samples=0, sample_rate=FS, max_dt_s=10,
                      second_sample=605, second_tolerance_samples=15,
                      second_dt_samples=300, second_dt_weight=5.0)
    ends = np.arange(100) * 10 + 9
    assert (t["dt_w"][(ends >= 605) & (ends < 605 + 15 + 300)] == 5.0).all()
    assert (t["dt_w"][ends < 605] == 1.0).all()
    assert (t["dt_w"][ends >= 605 + 15 + 300] == 1.0).all()


def test_a_traces_own_later_onsets_restart_or_mask_dt():
    """A picked later onset restarts dt like a second event; around an
    unpicked one (TauP only) dt is not trained."""
    t = token_targets(200, 10, p_sample=205, tolerance_samples=0, sample_rate=FS, max_dt_s=10,
                      later=[(605, 50, True), (1405, 100, False)],
                      mask_before_samples=100, mask_after_samples=1000)
    ends = np.arange(200) * 10 + 9
    assert (t["y"][ends >= 205] == 1).all()
    assert t["dt"][ends == 699][0] == pytest.approx(0.94)          # restarted at 605
    assert (t["dt_mask"][np.abs(ends - 605) < 50] == 0).all()
    masked = (ends >= 1405 - 200) & (ends < 1405 + 1100)
    assert (t["dt_mask"][masked] == 0).all()
    assert (t["dt_mask"][(ends >= 700) & (ends < 1405 - 200)] == 1).all()


def test_later_onsets_round_trip():
    from onset.later import decode, encode
    got = decode(encode([(3712.4, 0.5, "a"), (5000.0, 1.0, "m")]))
    assert got == [(3712.4, 0.5, "a"), (5000.0, 1.0, "m")]
    assert decode(float("nan")) == [] and encode([]) is None


def test_dataset_moves_later_onsets_with_p(tmp_path):
    """A stored trace's own later onsets follow P through the splice, and the
    picked ones are carried for evaluation."""
    from onset.data import OnsetDataset
    rng = np.random.default_rng(5)
    w = StoreWriter(tmp_path)
    sta = next(s for s in (f"S{i}" for i in range(100)) if station_split("XX", s) == "train")
    base = {"split": "train", "source": "fdsn", "network": "XX", "station": sta}
    w.add("context/1/XX.S", rng.normal(0, 1, (12000, 3)).astype(np.float32),
          np.zeros(12000, bool), {**base, "kind": "context"})
    ev = rng.normal(0, 1, (6000, 3)).astype(np.float32)
    ev[700:] *= 20
    ev[3700:] *= 5
    w.add("event/1/XX.S", ev, np.zeros(6000, bool),
          {**base, "kind": "event", "p_sample": 700.0, "p_source": "aic", "p_tolerance_s": 0.3,
           "context_key": "context/1/XX.S", "later_p": "3700.0:0.5:a;5000.0:1:m"})
    w.close()
    for lead in (0.0, 40.0):
        ds = OnsetDataset(tmp_path, "train", DataConfig(eval_lead_in_s=lead, eval_second_every=0),
                          ModelConfig(), False)
        b = ds[0]
        p_s = float(b["p_s"])
        later = [v for v in b["later_s"].tolist() if np.isfinite(v)]
        assert later == [pytest.approx(p_s + 30.0)]
        ends = (np.arange(len(b["y"])) * 10 + 9) / FS
        k = np.flatnonzero(ends >= p_s + 31.0)[0]
        assert float(b["dt"][k]) == pytest.approx(ends[k] - (p_s + 30.0), abs=1e-4)
        assert (b["dt_mask"].numpy()[(ends >= p_s + 43.0) & (ends < p_s + 53.0)] == 0).all()


def test_sweep_scores_later_onsets():
    stride, n = 10, 600
    p = np.zeros(n)
    p[100:] = 0.99
    dt = np.minimum(10.0, np.maximum(0.0, 0.1 * (np.arange(n) - 100)))
    dt[400:] = np.minimum(10.0, 0.1 * np.arange(200))  # restart at 40 s
    e = {"p": p, "dt": dt, "p_s": 10.0, "tol_s": 0.3, "later_s": [40.0]}
    row = metrics.sweep([e], [], stride, FS, thresholds=[0.9], rule=metrics.TriggerRule())[0]
    assert row["later_n"] == 1 and row["later_recall@1.0s"] == 1.0


def token_trigger_reference(p, dt, thr, rel, below, frm, ntok):
    """Token by token, as ayzek's src/pipeline/trigger.hpp."""
    out, armed, peak, low = [], True, 0.0, 0
    for i in range(len(p)):
        if p[i] < rel:
            armed, peak, low = True, 0.0, 0
            continue
        if p[i] < thr:
            peak, low = max(peak, dt[i]), 0
            continue
        fire = False
        if armed:
            fire = True
        elif peak >= frm and dt[i] <= below:
            low += 1
            fire = low >= ntok
        else:
            low = 0
        armed = False
        if fire:
            out.append(i)
            peak, low = dt[i], 0
        else:
            peak = max(peak, dt[i])
    return out


def test_trigger_tokens_is_ayzeks_trigger():
    rng = np.random.default_rng(7)
    for seed in range(40):
        n = 600
        # Events: p high over stretches, dt a sawtooth that restarts at random.
        p = np.where(rng.random(n) < 0.02, 0.95, 0.1)
        p = np.convolve(p, np.ones(40), "same").clip(0, 1) + rng.normal(0, 0.02, n)
        dt = np.zeros(n)
        for i in range(1, n):
            dt[i] = 0.0 if rng.random() < 0.02 else min(10.0, dt[i - 1] + 0.1 + rng.normal(0, 0.3))
        for ntok in (1, 2, 3):
            rule = metrics.TriggerRule(True, 2.0, 5.0, ntok, min_gap_s=0.0)
            got = metrics.trigger_tokens(p, dt, 0.9, 0.45, 10, FS, rule).tolist()
            assert got == token_trigger_reference(p, dt, 0.9, 0.45, 2.0, 5.0, ntok), (seed, ntok)
        off = metrics.TriggerRule(False, min_gap_s=0.0)
        assert (metrics.trigger_tokens(p, dt, 0.9, 0.45, 10, FS, off).tolist()
                == metrics.rising_edges(p, 0.9, 0.45).tolist())


def test_trigger_gap_drops_close_p_dates():
    p = np.full(400, 0.99)
    p[0] = 0.0
    dt = np.minimum(10.0, 0.1 * np.arange(400))
    dt[150:] = np.minimum(10.0, 0.1 * np.arange(250))            # restart at 15 s
    dt[80:] = np.where(np.arange(80, 400) < 150, 0.0, dt[80:])  # and one at 8 s
    rule = metrics.TriggerRule(True, 2.0, 5.0, 2, min_gap_s=15.0)
    edges = metrics.trigger_tokens(p, dt, 0.9, 0.45, 10, FS, rule).tolist()
    t = metrics.token_times(400, 10, FS)
    dates = t[edges] - dt[edges]
    assert edges[0] == 1 and np.all(np.diff(dates) >= 15.0)


def test_sweep_scores_second_onsets():
    stride, n = 10, 600
    p = np.zeros(n)
    p[100:] = 0.99                                  # first onset at 10 s, never releases
    dt = np.minimum(10.0, np.maximum(0.0, 0.1 * (np.arange(n) - 100)))
    dt[400:] = np.minimum(10.0, 0.1 * np.arange(200))  # restart at 40 s
    e = {"p": p, "dt": dt, "p_s": 10.0, "tol_s": 0.3, "p2_s": 40.0, "tol2_s": 0.3}
    rows_on = metrics.sweep([e], [], stride, FS, thresholds=[0.9], rule=metrics.TriggerRule())
    rows_off = metrics.sweep([e], [], stride, FS, thresholds=[0.9])
    assert rows_on[0]["second_n"] == 1 and rows_on[0]["second_recall@1.0s"] == 1.0
    assert rows_off[0]["second_recall@1.0s"] == 0.0
    assert rows_on[0]["onset_recall@1.0s"] == 1.0 and rows_off[0]["onset_recall@1.0s"] == 0.5


@pytest.fixture
def two_event_store(tmp_path):
    rng = np.random.default_rng(5)
    w = StoreWriter(tmp_path)
    sta = next(s for s in (f"S{i}" for i in range(100)) if station_split("XX", s) == "train")
    base = {"split": "train", "source": "fdsn", "network": "XX", "station": sta}
    w.add("context/1/XX.S", rng.normal(0, 1, (12000, 3)).astype(np.float32),
          np.zeros(12000, bool), {**base, "kind": "context"})
    for k, amp in ((1, 20.0), (2, 5.0)):
        ev = rng.normal(0, 1, (6000, 3)).astype(np.float32)
        ev[700:] *= amp * np.exp(-np.arange(5300) / 1500)[:, None] + 1
        w.add(f"event/{k}/XX.S", ev, np.zeros(6000, bool),
              {**base, "kind": "event", "p_sample": 700.0, "p_source": "aic",
               "p_tolerance_s": 0.3, "context_key": "context/1/XX.S", "magnitude": 3.0})
    w.close()
    return tmp_path


def test_second_event_goes_into_the_coda(two_event_store):
    from onset.data import OnsetDataset
    dc = DataConfig(seq_seconds=40, gap_aug_p=0, lead_in_p=0, second_p=1.0)
    ds = OnsetDataset(two_event_store, "train", dc, ModelConfig(), True)
    seen = 0
    for seed in range(20):
        ds.reseed(seed)
        b = ds[0]
        if not np.isfinite(float(b["p2_s"])):
            continue
        seen += 1
        p1, p2 = float(b["p_s"]), float(b["p2_s"])
        assert dc.second_min_s - 1e-6 <= p2 - p1 <= dc.second_max_s + 1e-6
        ends = (np.arange(len(b["y"])) * 10 + 9) / FS
        after = ends >= p2 + 0.5
        assert (b["y"].numpy()[ends >= p1 + 0.5] == 1).all()
        assert np.allclose(b["dt"].numpy()[after], np.minimum(10, ends[after] - p2), atol=1e-4)
        assert (b["geo_mask"].numpy()[ends >= p2 - 0.3] == 0).all()
        # the added event raises the level at its P
        x = b["x"].numpy()[:, :3]
        k = int(p2 * FS)
        assert np.abs(x[k:k + 200]).mean() > np.abs(x[k - 250:k - 50]).mean()
    assert seen >= 10


def test_evaluation_adds_second_onsets_to_every_nth_event(two_event_store):
    from onset.data import OnsetDataset
    ds = OnsetDataset(two_event_store, "train", DataConfig(eval_lead_in_s=0, eval_second_every=2),
                      ModelConfig(), False)
    ev = [i for i, k in enumerate(ds.rows.kind) if k == "event"]
    got = [np.isfinite(float(ds[i]["p2_s"])) for i in ev]
    assert got == [i % 2 == 1 for i in ev]
    again = OnsetDataset(two_event_store, "train", DataConfig(eval_lead_in_s=0, eval_second_every=2),
                         ModelConfig(), False)
    k = ev[1]
    assert float(again[k]["p2_s"]) == float(ds[k]["p2_s"])        # a fixed draw


# -- noise contamination -------------------------------------------------------

def test_coda_duration_grows_with_magnitude():
    from onset.catalog import coda_seconds
    assert coda_seconds(2.0) == pytest.approx(27.2, abs=0.5)
    assert coda_seconds(3.0) == pytest.approx(86.1, abs=0.5)
    assert coda_seconds(5.0) == pytest.approx(860.9, abs=1.0)
    assert coda_seconds(8.0) == 3600.0                      # capped
    assert coda_seconds(8.0, cap=600.0) == 600.0


def test_noise_in_a_coda_is_contaminated(tmp_path):
    """A P just before the window is not in it, but its coda is."""
    pytest.importorskip("obspy")
    from onset.catalog import Catalog, TravelTimes
    csv = tmp_path / "cat.csv"
    csv.write_text("Date,Longitude,Latitude,Depth,Rms,Type,Magnitude,Location,EventID\n"
                   "01/06/2025 12:00:00,28.00,39.20,10,0.1,ML,3.0,A,1\n"      # 22 km from the station
                   "01/06/2025 12:30:00,31.50,39.20,10,0.1,ML,2.0,B,2\n",     # 300 km, M2: not visible
                   encoding="utf-8")
    cat, taup = Catalog(csv), TravelTimes()
    lat, lon = 39.0, 28.0
    origin = cat.event(1).origin
    p = origin + taup.first(22.2, 10.0)
    # A window starting 30 s after the P: no arrival inside, but in the M3's
    # ~86 s coda.
    assert cat.arrivals(lat, lon, p + 30, p + 150, taup) == []
    assert [e for e, _ in cat.ringing(lat, lon, p + 30, p + 150, taup)] == [1]
    assert cat.ringing(lat, lon, p + 30, p + 150, taup, coda_cap=0.0) == []
    # Once the coda is over, the window is clean.
    assert cat.ringing(lat, lon, p + 120, p + 240, taup) == []
    # A P inside the window is caught either way.
    assert [e for e, _ in cat.ringing(lat, lon, p - 10, p + 10, taup, coda_cap=0.0)] == [1]
    # A distant small event is not visible, coda or not.
    o2 = cat.event(2).origin
    assert cat.ringing(lat, lon, o2, o2 + 300, taup) == []


def test_training_stops_without_noise(tmp_path):
    """A store with events and no noise cannot train or validate a detector."""
    from onset.config import TrainConfig
    from onset.train import check_noise
    rng = np.random.default_rng(6)
    w = StoreWriter(tmp_path)
    for split in ("train", "val"):
        sta = next(s for s in (f"S{i}" for i in range(200)) if station_split("XX", s) == split)
        w.add(f"event/1/XX.{sta}", rng.normal(0, 1, (6000, 3)).astype(np.float32),
              np.zeros(6000, bool),
              {"split": split, "source": "fdsn", "network": "XX", "station": sta,
               "kind": "event", "p_sample": 700.0, "p_source": "aic", "p_tolerance_s": 0.3})
    w.close({"dropped": {"noise_absent": 2, "context_absent": 2}})
    with pytest.raises(SystemExit, match="no training noise.*no validation noise"):
        check_noise(tmp_path, TrainConfig())


def test_training_warns_about_little_validation_noise(tmp_path, capsys):
    from onset.config import TrainConfig
    from onset.train import check_noise
    rng = np.random.default_rng(7)
    w = StoreWriter(tmp_path)
    for split, n in (("train", 60000), ("val", 60000)):          # 10 min each
        sta = next(s for s in (f"S{i}" for i in range(200)) if station_split("XX", s) == split)
        base = {"split": split, "source": "fdsn", "network": "XX", "station": sta}
        w.add(f"noise/1/XX.{sta}", rng.normal(0, 1, (n, 3)).astype(np.float32),
              np.zeros(n, bool), {**base, "kind": "noise"})
        w.add(f"event/1/XX.{sta}", rng.normal(0, 1, (6000, 3)).astype(np.float32),
              np.zeros(6000, bool), {**base, "kind": "event", "p_sample": 700.0,
                                     "p_source": "aic", "p_tolerance_s": 0.3})
    w.close()
    check_noise(tmp_path, TrainConfig())
    out = capsys.readouterr().out
    assert "val 1 traces (0.2 h)" in out
    assert "one false trigger is 6.00/h" in out


def test_exact_operating_point_lands_between_grid_steps():
    """Noise peaks spread between two grid thresholds: the exact point meets
    the budget with a threshold the grid does not have."""
    rng = np.random.default_rng(8)
    stride, n = 10, 36000                                   # 1 h per trace
    noise = []
    for _ in range(10):
        p = np.zeros(n)
        peaks = 0.9820 + 0.0035 * rng.random(3)             # between two grid steps
        for k, v in zip(rng.choice(np.arange(100, n - 100, 400), 3, replace=False), peaks):
            p[k] = v
        noise.append({"p": p, "dt": np.zeros(n), "missing_tokens": np.zeros(n, bool)})
    ev_p = np.zeros(600)
    ev_p[100:] = 0.984
    events = [{"p": ev_p, "dt": np.zeros(600), "p_s": 10.0, "tol_s": 0.3}]
    rows = metrics.sweep(events, noise, stride, FS)
    grid = metrics.operating_point(rows, 1.0)
    exact = metrics.exact_operating_point(rows, events, noise, stride, FS, 1.0)
    assert exact["false_per_hour"] <= 1.0 < rows[[r["threshold"] for r in rows].index(
        max(r["threshold"] for r in rows if r["threshold"] < grid["threshold"]))]["false_per_hour"]
    assert exact["threshold"] < grid["threshold"]
    assert exact["recall@1.0s"] == 1.0 and grid["recall@1.0s"] == 0.0


def test_noise_audit_finds_triggers_seen_at_several_stations():
    import pandas as pd
    from onset.audit import coincidences
    # Pull 1: an event at A and B 3 s apart (stations 20 km apart); pull 2:
    # blips at C and D a minute apart; A's second blip alone.
    trig = pd.DataFrame({
        "pull": [1, 1, 1, 2, 2],
        "station": ["A", "B", "A", "C", "D"],
        "t": [100.0, 103.0, 250.0, 100.0, 160.0],
        "lat": [39.0, 39.18, 39.0, 40.0, 40.1],
        "lon": [28.0, 28.0, 28.0, 29.0, 29.0],
    })
    assert coincidences(trig).tolist() == [True, True, False, False, False]
    # Sliding B by 100 s around its 290 s window breaks the pair.
    period = {(1, "A"): (0.0, 290.0), (1, "B"): (0.0, 290.0)}
    assert not coincidences(trig, shifts={(1, "B"): 100.0}, period=period).any()


def _catalog(tmp_path, rows):
    csv = tmp_path / "cat.csv"
    csv.write_text("Date,Longitude,Latitude,Depth,Rms,Type,Magnitude,Location,EventID\n"
                   + "".join(f"{d},{lon},{lat},10,0.1,ML,{m},X,{i}\n"
                             for i, (d, lat, lon, m) in enumerate(rows, 1)), encoding="utf-8")
    from onset.catalog import Catalog
    return Catalog(csv)


def test_catalogue_activity_and_aftermath(tmp_path):
    cat = _catalog(tmp_path, [
        ("27/10/2025 19:48:00", 39.20, 28.20, 6.1),     # a mainshock
        ("28/10/2025 01:00:00", 39.21, 28.21, 2.0),
        ("28/10/2025 02:00:00", 39.22, 28.19, 2.2),
        ("28/10/2025 03:00:00", 39.19, 28.22, 1.8),
        ("01/06/2026 12:00:00", 37.00, 36.00, 5.0),     # far away, later
    ])
    t = pd.Timestamp("2025-10-28 02:30", tz="UTC").timestamp()
    assert cat.nearby_count(39.08, 28.98, t - 43200, t + 43200, 75.0) == 4
    assert cat.nearby_count(40.60, 27.70, t - 43200, t + 43200, 75.0) == 0
    # M6.1: 30 d x 10^(1.1/2) = 106 days of aftermath within 150 km.
    assert cat.aftermath_of(39.08, 28.98, t, 150.0, 5.0, 30.0) == (1, 6.1)
    later = pd.Timestamp("2026-03-01", tz="UTC").timestamp()           # 125 days on
    assert cat.aftermath_of(39.08, 28.98, later, 150.0, 5.0, 30.0) is None
    assert cat.aftermath_of(41.50, 28.98, t, 150.0, 5.0, 30.0) is None  # 250 km away


def test_noise_rules_and_exclusions(tmp_path):
    from onset import build_fdsn
    from collections import Counter
    audit = tmp_path / "audit.csv"
    pd.DataFrame({"pull": [7, 7, 7, 9], "station": ["A", "B", "C", "D"],
                  "key": ["noise/7/KO.A", "noise/7/KO.B", "noise/7/KO.C", "noise/9/KO.D"],
                  "coincident": [True, True, True, False]}).to_csv(audit, index=False)
    keys, pulls = build_fdsn.load_exclusions([audit])
    assert keys == {"noise/7/KO.A", "noise/7/KO.B", "noise/7/KO.C"} and pulls == {7}
    cat = _catalog(tmp_path, [("28/10/2025 01:00:00", 39.21, 28.21, 2.0),
                              ("28/10/2025 02:00:00", 39.22, 28.19, 2.2),
                              ("28/10/2025 03:00:00", 39.19, 28.22, 1.8),
                              ("28/10/2025 04:00:00", 39.20, 28.20, 2.4)])
    build_fdsn._W.update(catalog=cat, cfg={
        "exclude_keys": keys, "exclude_pulls": pulls, "max_active_events": 3,
        "active_radius_km": 75.0, "active_hours": 12.0, "aftermath_mag": 5.0,
        "aftermath_radius_km": 150.0, "aftermath_days": 30.0})
    t = pd.Timestamp("2025-10-28 02:30", tz="UTC").timestamp()
    quiet = pd.Timestamp("2025-06-01", tz="UTC").timestamp()
    drops = Counter()
    ok = lambda ev, sta, lat, lon, t0: build_fdsn._noise_vouched(ev, "KO", sta, lat, lon, t0, 290, drops)
    assert not ok(7, "Z", 40.6, 27.7, quiet)            # a whole pull with a 3-station event
    assert ok(9, "D", 40.6, 27.7, quiet)                # pull 9 had no coincidence
    assert not ok(1, "S", 39.08, 28.98, t)              # 4 events within 75 km in +-12 h
    assert ok(1, "S", 40.60, 27.70, t)                  # the same time, far away
    assert drops == Counter({"noise_excluded": 1, "noise_active": 1})


# -- run directories -------------------------------------------------------------

def test_a_run_directory_is_never_reused(tmp_path):
    from onset.train import prepare_out
    assert prepare_out(tmp_path / "new", False, False) is None
    run = tmp_path / "run"
    run.mkdir()
    (run / "history.jsonl").write_text("{}\n")
    (run / "best.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="already holds a run"):
        prepare_out(run, False, False)
    with pytest.raises(SystemExit, match="no state.pt"):
        prepare_out(run, True, False)
    assert prepare_out(run, False, True) is None
    moved = [d for d in run.iterdir() if d.name.startswith("previous_")]
    assert len(moved) == 1 and (moved[0] / "history.jsonl").exists()
    assert not (run / "history.jsonl").exists()


def test_resume_reads_the_saved_state(tmp_path):
    import torch
    from onset.train import prepare_out
    torch.save({"model": {}, "opt": {}, "epoch": 7, "step": 7000, "best": 0.5},
               tmp_path / "state.pt")
    st = prepare_out(tmp_path, True, False)
    assert st["epoch"] == 7 and st["step"] == 7000 and st["best"] == 0.5
