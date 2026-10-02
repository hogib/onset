"""Peak P displacement (pd.py) and the magnitude relation (pd_fit.py)."""
import numpy as np
import pandas as pd
import pytest

from onset import pd_fit
from onset.pd import FS, displacement, peak_displacement


def test_displacement_of_a_sinusoid():
    """2 Hz velocity of amplitude A m/s is displacement of amplitude
    A / (2 pi 2 Hz), well above the 0.075 Hz high-pass."""
    t = np.arange(int(60 * FS)) / FS
    a, f, gain = 1e-6, 2.0, 1.5e9
    d = displacement(gain * a * np.sin(2 * np.pi * f * t), gain, "m/s")
    assert np.abs(d[-int(10 * FS):]).max() == pytest.approx(a / (2 * np.pi * f), rel=0.02)


def test_displacement_is_causal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, int(40 * FS))
    y = x.copy()
    y[3000:] += 50.0 * rng.normal(0, 1, len(y) - 3000)
    assert np.array_equal(displacement(x, 1.0, "m/s")[:3000], displacement(y, 1.0, "m/s")[:3000])


def test_peak_displacement_windows():
    d = np.zeros(int(40 * FS))
    p = int(20 * FS)
    d[p + 50] = 1.0                       # 0.5 s after P
    d[p + 250] = -3.0                     # 2.5 s after P
    d[p - 500] = 0.2                      # in the noise window
    got = peak_displacement(d, p)
    assert got["pd_1s"] == 1.0 and got["pd_2s"] == 1.0 and got["pd_3s"] == 3.0
    assert got["pd_noise"] == 0.2


ALPHA, BETA, GAMMA = -5.0, 0.75, -1.5


def synthetic(n_events=600, seed=0, scatter=0.3, noise=1e-12):
    """Pd from a known relation with station terms, M uniform on 2-6; values
    below MIN_SNR x `noise` are censored."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 0.2, 30)
    terms = {f"S{i}": t for i, t in enumerate(raw - raw.mean())}   # sum zero, as fitted
    rows = []
    for j in range(n_events):
        m = rng.uniform(2, 6)
        for s in rng.choice(list(terms), 8, replace=False):
            dist = rng.uniform(10, 150)
            lp = ALPHA + BETA * m + GAMMA * pd_fit.log_r(dist) + terms[s] + rng.normal(0, scatter)
            rows.append({"event_id": j, "station": s, "magnitude": m, "distance_km": dist,
                         "split": "train", "pd_noise": noise,
                         **{f"pd_{t:g}s": 10 ** lp for t in pd_fit.PD_WINDOWS_S}})
    return pd.DataFrame(rows), terms


def test_censored_fit_recovers_the_relation_and_station_terms():
    df, terms = synthetic()
    f = pd_fit.fit_censored(df, 3.0, knots=(), m_knots=())   # the synthetic relation is linear
    assert f["converged"]
    assert f["beta"][0] == pytest.approx(BETA, abs=0.03)
    assert f["gamma"][0] == pytest.approx(GAMMA, abs=0.1)
    assert f["sigma"] == pytest.approx(0.3, abs=0.03)
    got = np.array([f["station"][s] for s in terms])
    assert np.corrcoef(got, list(terms.values()))[0, 1] > 0.9


def test_distance_hinges_follow_a_bend_in_attenuation():
    """Attenuation that steepens beyond 70 km is followed by the hinge."""
    rng = np.random.default_rng(1)
    rows = []
    for j in range(400):
        m = rng.uniform(2, 6)
        for k in range(8):
            dist = rng.uniform(10, 200)
            lr = pd_fit.log_r(dist)
            lp = ALPHA + BETA * m + GAMMA * lr - 1.0 * max(0.0, lr - np.log10(70.0))
            rows.append({"event_id": j, "station": f"S{k}", "magnitude": m, "distance_km": dist,
                         "split": "train", "pd_noise": 1e-12,
                         **{f"pd_{t:g}s": 10 ** (lp + rng.normal(0, 0.2)) for t in pd_fit.PD_WINDOWS_S}})
    f = pd_fit.fit_censored(pd.DataFrame(rows), 3.0, knots=(70.0,), m_knots=())
    assert f["gamma"][1] == pytest.approx(-1.0, abs=0.2)
    assert f["beta"][0] == pytest.approx(BETA, abs=0.03)


def test_magnitude_hinge_follows_a_steeper_slope_above_the_knot():
    rng = np.random.default_rng(2)
    rows = []
    for j in range(800):
        m = rng.uniform(2, 6)
        for k in range(8):
            dist = rng.uniform(10, 150)
            lp = ALPHA + BETA * m + 0.4 * max(0.0, m - 4.0) + GAMMA * pd_fit.log_r(dist)
            rows.append({"event_id": j, "station": f"S{k}", "magnitude": m, "distance_km": dist,
                         "split": "train", "pd_noise": 1e-12,
                         **{f"pd_{t:g}s": 10 ** (lp + rng.normal(0, 0.2)) for t in pd_fit.PD_WINDOWS_S}})
    df = pd.DataFrame(rows)
    f = pd_fit.fit_censored(df, 3.0, knots=(), m_knots=(4.0,))
    assert f["beta"] == [pytest.approx(BETA, abs=0.03), pytest.approx(0.4, abs=0.06)]
    big = df[df.magnitude >= 5.5]
    est = pd_fit.estimate_censored(big, f)
    assert abs((est.est - est.magnitude).mean()) < 0.05


def test_censoring_flattens_least_squares_but_not_the_censored_fit():
    """A noise floor that hides most small events biases least squares on the
    values above it (the slope flattens); the censored fit is unaffected."""
    df, _ = synthetic(noise=10 ** (ALPHA + BETA * 4.0 + GAMMA * 2.0) / pd_fit.MIN_SNR)
    assert pd_fit.usable(df, 3.0).mean() < 0.6
    ols, cen = pd_fit.fit_ols(df, 3.0, knots=()), pd_fit.fit_censored(df, 3.0, knots=(), m_knots=())
    assert ols["beta"] < BETA - 0.1
    assert cen["beta"][0] == pytest.approx(BETA, abs=0.05)


def test_large_events_are_not_shrunk():
    """Regressing M on log Pd pulls the largest events towards the mean; the
    censored relation, maximised per event, does not."""
    df, _ = synthetic(scatter=0.5)
    big = df[df.magnitude >= 5.5]
    cen = pd_fit.estimate_censored(big, pd_fit.fit_censored(df, 3.0, knots=(), m_knots=()))
    inv = pd_fit.estimate_median(big, pd_fit.fit_ols(df, 3.0, inverse=True, knots=()), inverse=True)
    assert abs((cen.est - cen.magnitude).mean()) < 0.1
    assert (inv.est - inv.magnitude).mean() < -0.2


def test_gutenberg_richter_prior_corrects_the_selection_of_small_events():
    """Small events are estimated only when some station records them above
    the noise, which selects the ones that came out bright: the
    maximum-likelihood estimate is then biased high. The posterior under the
    population's Gutenberg-Richter distribution removes most of that bias."""
    rng = np.random.default_rng(3)
    b = 1.0
    m_all = 1.5 - np.log10(rng.uniform(size=4000)) / b             # GR above M1.5
    noise = 10 ** (ALPHA + BETA * 3.5 + GAMMA * 2.0) / pd_fit.MIN_SNR
    rows = []
    for j, m in enumerate(m_all[m_all < 6.5]):
        for k in range(6):
            dist = rng.uniform(20, 150)
            lp = ALPHA + BETA * m + GAMMA * pd_fit.log_r(dist) + rng.normal(0, 0.3)
            rows.append({"event_id": j, "station": f"S{k}", "magnitude": m, "distance_km": dist,
                         "split": "train", "pd_noise": noise,
                         **{f"pd_{t:g}s": 10 ** lp for t in pd_fit.PD_WINDOWS_S}})
    df = pd.DataFrame(rows)
    f = {"tau_s": 3.0, "alpha": ALPHA, "beta": [BETA], "m_knots": [], "gamma": [GAMMA],
         "knots_km": [], "sigma": 0.3, "station": {}}
    small = df[df.magnitude < 3.0]
    ml = pd_fit.estimate_censored(small, f, b=0.0)
    post = pd_fit.estimate_censored(small, f, b=b)
    assert len(ml) > 50
    bias_ml, bias_post = (ml.est - ml.magnitude).mean(), (post.est - post.magnitude).mean()
    assert bias_ml > 0.15
    assert abs(bias_post) < bias_ml / 2
    assert (post.sd > 0).all()


def test_a_channel_that_does_not_record_is_left_out():
    """A station whose pre-P noise is far below its usual level reports every
    value as censored; its spurious upper bound would pull the estimate down."""
    f = {"tau_s": 3.0, "alpha": ALPHA, "beta": [BETA], "m_knots": [], "gamma": [GAMMA],
         "knots_km": [], "sigma": 0.3, "station": {}, "b_value": 0.0,
         "station_noise": {"A": 1e-7, "B": 1e-7, "DEAD": 1e-7}}
    m = 5.0
    rows = [{"event_id": 0, "station": s, "distance_km": 40.0, "magnitude": m,
             "pd_noise": 1e-7, "pd_3s": 10 ** (ALPHA + BETA * m + GAMMA * pd_fit.log_r(40.0))}
            for s in ("A", "B")]
    rows.append({"event_id": 0, "station": "DEAD", "distance_km": 40.0, "magnitude": m,
                 "pd_noise": 2e-9, "pd_3s": 3e-9})        # 50x below its median
    est = pd_fit.estimate_censored(pd.DataFrame(rows), f)
    assert est.est.iloc[0] == pytest.approx(m, abs=0.05)
    f["station_noise"] = {}                                   # without the guard
    biased = pd_fit.estimate_censored(pd.DataFrame(rows), f)
    assert biased.est.iloc[0] < m - 0.3
