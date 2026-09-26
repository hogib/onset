"""The geometry locator (`onset.locate`), the reference for ayzek's.

The same scenarios are in ayzek's tests/test_locate.cpp, so the two
implementations are held to the same answers.
"""
import numpy as np
import pytest

from onset.locate import (StationEstimate, azimuth_rad, destination, distance_km, locate,
                          locate_robust)

EPI = (39.20, 28.10)               # an epicentre near Sındırgı
ORIGIN = 1000.0
VP, DEPTH = 6.0, 10.0
STATIONS = {"A": (39.60, 28.00), "B": (38.90, 27.60), "C": (39.10, 28.70), "D": (39.45, 28.55)}


def observe(code, dist_sd=0.1, kappa=20.0, dp=0.0, dist_scale=1.0, baz_off_deg=0.0):
    lat, lon = STATIONS[code]
    d = float(distance_km(lat, lon, *EPI))
    return StationEstimate(code, lat, lon, ORIGIN + np.hypot(d, DEPTH) / VP + dp,
                           np.log(d * dist_scale), dist_sd,
                           float(azimuth_rad(lat, lon, *EPI)) + np.radians(baz_off_deg), kappa)


def test_geodesy_round_trips():
    lat, lon = destination(39.0, 28.0, np.radians(60.0), 50.0)
    assert distance_km(39.0, 28.0, lat, lon) == pytest.approx(50.0, rel=1e-6)
    assert np.degrees(azimuth_rad(39.0, 28.0, lat, lon)) == pytest.approx(60.0, abs=1e-6)


def test_exact_observations_recover_the_epicentre():
    loc = locate([observe(c) for c in STATIONS])
    assert distance_km(loc.lat, loc.lon, *EPI) < 1.0
    assert loc.origin == pytest.approx(ORIGIN, abs=0.1)
    assert loc.rms < 0.05
    assert loc.n_stations == 4


def test_one_station_with_a_back_azimuth_is_enough():
    loc = locate([observe("A")])
    assert distance_km(loc.lat, loc.lon, *EPI) < 1.0


def test_one_station_without_a_back_azimuth_is_not():
    o = observe("A")
    o.baz_rad = np.nan
    assert locate([o]) is None


def test_distance_and_time_alone_locate_three_stations():
    obs = [observe(c, kappa=0.0) for c in ("A", "B", "C")]
    loc = locate(obs)
    assert distance_km(loc.lat, loc.lon, *EPI) < 2.0


def test_uncertain_stations_count_for_less():
    """A station 30% off in distance and 40 deg off in direction moves the
    solution little when it says it is unsure, and a lot when it says it is sure."""
    good = [observe(c) for c in ("A", "B", "C")]
    unsure = locate(good + [observe("D", dist_sd=1.0, kappa=0.5, dist_scale=1.3, baz_off_deg=40)])
    sure = locate(good + [observe("D", dist_sd=0.02, kappa=200, dist_scale=1.3, baz_off_deg=40)])
    assert distance_km(unsure.lat, unsure.lon, *EPI) < 2.0
    assert distance_km(sure.lat, sure.lon, *EPI) > distance_km(unsure.lat, unsure.lon, *EPI) + 2.0


def test_a_bad_p_time_is_dropped():
    obs = [observe(c) for c in ("A", "B", "C")] + [observe("D", dp=8.0)]
    loc = locate_robust(obs, max_rms=1.0)
    assert loc.dropped == ["D"]
    assert distance_km(loc.lat, loc.lon, *EPI) < 1.0


def test_error_radius_follows_the_stated_uncertainty():
    tight = locate([observe("A", dist_sd=0.05, kappa=100)])
    loose = locate([observe("A", dist_sd=0.4, kappa=5)])
    assert loose.err_km > 2 * tight.err_km
