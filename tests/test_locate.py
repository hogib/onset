"""The geometry locator (`onset.locate`), the reference for ayzek's.

The same scenarios are in ayzek's tests/test_locate.cpp, so the two
implementations are held to the same answers.
"""
import numpy as np
import pytest

from onset.locate import StationEstimate, distance_km, locate, locate_robust

EPI = (39.20, 28.10)               # an epicentre near Sındırgı
ORIGIN = 1000.0
VP, DEPTH = 6.0, 10.0
STATIONS = {"A": (39.60, 28.00), "B": (38.90, 27.60), "C": (39.10, 28.70), "D": (39.45, 28.55)}


def observe(code, dist_sd=0.1, dp=0.0, dist_scale=1.0):
    lat, lon = STATIONS[code]
    d = float(distance_km(lat, lon, *EPI))
    return StationEstimate(code, lat, lon, ORIGIN + np.hypot(d, DEPTH) / VP + dp,
                           np.log(d * dist_scale), dist_sd)


def test_one_degree_of_latitude():
    assert distance_km(39.0, 28.0, 40.0, 28.0) == pytest.approx(111.19, abs=0.01)


def test_exact_observations_recover_the_epicentre():
    loc = locate([observe(c) for c in STATIONS])
    assert distance_km(loc.lat, loc.lon, *EPI) < 1.0
    assert loc.origin == pytest.approx(ORIGIN, abs=0.1)
    assert loc.rms < 0.05
    assert loc.n_stations == 4


def test_two_stations_are_not_enough():
    """Two rings cross in two mirror points."""
    assert locate([observe("A"), observe("B")]) is None


def test_three_stations_are_enough():
    loc = locate([observe(c) for c in ("A", "B", "C")])
    assert distance_km(loc.lat, loc.lon, *EPI) < 1.0


def test_uncertain_stations_count_for_less():
    """A station 50% off in distance moves the solution little when it says it
    is unsure, and a lot when it says it is sure."""
    good = [observe(c) for c in ("A", "B", "C")]
    unsure = locate(good + [observe("D", dist_sd=1.0, dist_scale=1.5)])
    sure = locate(good + [observe("D", dist_sd=0.02, dist_scale=1.5)])
    assert distance_km(unsure.lat, unsure.lon, *EPI) < 2.0
    assert distance_km(sure.lat, sure.lon, *EPI) > distance_km(unsure.lat, unsure.lon, *EPI) + 2.0


def test_a_bad_p_time_is_dropped():
    obs = [observe(c) for c in ("A", "B", "C")] + [observe("D", dp=8.0)]
    loc = locate_robust(obs, max_rms=1.0)
    assert loc.dropped == ["D"]
    assert distance_km(loc.lat, loc.lon, *EPI) < 1.0


def test_error_radius_follows_the_stated_uncertainty():
    """With loose P times, the rings' widths set the error radius."""
    tight = locate([observe(c, dist_sd=0.05) for c in ("A", "B", "C")], sigma_p=10.0)
    loose = locate([observe(c, dist_sd=0.4) for c in ("A", "B", "C")], sigma_p=10.0)
    assert loose.err_km > 2 * tight.err_km
