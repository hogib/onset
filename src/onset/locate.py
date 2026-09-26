"""Locating an event from the geometry head, the reference for ayzek's locator.

The geometry head (`OnsetDetector.geometry`, `ModelConfig.geometry=1`) says,
at every token after P, where the event is as seen from one station:

    log_dist, log_dist_var    epicentral distance, Gaussian in log km
                              (log_dist_var is the log of its variance, sd_i^2)
    baz, kappa                direction station -> event, von Mises

The head is trained with exactly those negative log-likelihoods
(`train.geometry_loss`), so each station's output is a likelihood over the
epicentre, and a network location is the epicentre that maximises their
product, together with the P times the detector dates from dt:

    J(x) = sum_i  (log max(D_i(x), 1) - log_dist_i)^2 / (2 sd_i^2)
         + sum_i  kappa_i (1 - cos(AZ_i(x) - baz_i))
         + sum_i  (r_i(x) - origin(x))^2 / (2 sigma_p^2)

with D_i and AZ_i the distance and azimuth from station i to x, r_i the P
time minus the travel time from x in a uniform half-space, and origin(x)
their mean (the least-squares origin). A station with no back-azimuth (a
Z12 instrument, or a model whose direction is not yet confident) adds only
its distance and P time.

Unlike the S-P locator this needs no S pick and no 60 s picker window: it is
available from the trigger on, sharpens as the stations' own uncertainties
shrink (once S is inside the model's lookback), and one station with a
back-azimuth already gives an epicentre.

The search is a grid over the epicentre at fixed depth: +-3 deg at 0.05 deg,
then +-0.1 deg at 0.005 deg around the best point. ayzek's
`pipeline::locate_geometry` is a transcription of `locate`, and
`tests/test_locate.py` fixes the behaviour both must have.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EARTH_RADIUS_KM = 6371.0
# Delta J for a 68% region in two dimensions (chi-square 2.30 / 2).
ONE_SIGMA_DJ = 1.15


@dataclass
class StationEstimate:
    """One station's geometry output at some token after P, and its P time."""
    station: str
    lat: float
    lon: float
    p_time: float                   # seconds, any common epoch
    log_dist: float                 # log km
    log_dist_sd: float              # standard deviation of log_dist
    baz_rad: float = np.nan         # station -> event, clockwise from north; nan if unknown
    kappa: float = 0.0              # von Mises concentration of baz_rad


@dataclass
class Location:
    lat: float
    lon: float
    origin: float
    rms: float                      # P-time residual rms, seconds
    misfit: float                   # J at the solution
    err_km: float                   # radius of the 68% region on the coarse grid
    n_stations: int
    dropped: list = field(default_factory=list)


def estimates_from_head(out: dict, token: int | slice = -1, batch: int = 0) -> dict:
    """The geometry outputs of `OnsetDetector.forward` at one token as plain
    numbers: `log_dist`, `log_dist_sd`, `baz_rad` and `kappa`. The head's
    `log_dist_var` is the log of the variance."""
    v = out["baz_vec"][batch, token].float()
    return {"log_dist": float(out["log_dist"][batch, token]),
            "log_dist_sd": float((0.5 * out["log_dist_var"][batch, token]).exp()),
            "baz_rad": float(np.arctan2(float(v[0]), float(v[1]))),
            "kappa": float(out["baz_log_kappa"][batch, token].exp())}


def distance_km(lat1, lon1, lat2, lon2):
    """Haversine distance, as ayzek's `pipeline::distance_km`."""
    r = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * r / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def azimuth_rad(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, clockwise from north, radians."""
    r = np.pi / 180.0
    p1, p2, dl = lat1 * r, lat2 * r, (lon2 - lon1) * r
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def destination(lat, lon, az_rad, km):
    """The point `km` from (lat, lon) along bearing `az_rad`."""
    r = np.pi / 180.0
    d = km / EARTH_RADIUS_KM
    p1, l1 = lat * r, lon * r
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(az_rad))
    l2 = l1 + np.arctan2(np.sin(az_rad) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return p2 / r, l2 / r


def terms(lat, lon, obs: list[StationEstimate], vp: float, depth_km: float,
          sigma_p: float):
    """Per-station contributions to J at epicentres (lat, lon), any shape.

    Returns (J (n_st, ...), origin (...), P residuals (n_st, ...)).
    """
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    parts, resid = [], []
    for o in obs:
        d = distance_km(o.lat, o.lon, lat, lon)
        part = (np.log(np.maximum(d, 1.0)) - o.log_dist) ** 2 / (2.0 * o.log_dist_sd ** 2)
        if np.isfinite(o.baz_rad) and o.kappa > 0:
            part = part + o.kappa * (1.0 - np.cos(azimuth_rad(o.lat, o.lon, lat, lon) - o.baz_rad))
        parts.append(part)
        resid.append(o.p_time - np.hypot(d, depth_km) / vp)
    parts, resid = np.stack(parts), np.stack(resid)
    origin = resid.mean(axis=0)
    parts = parts + (resid - origin) ** 2 / (2.0 * sigma_p ** 2)
    return parts, origin, resid - origin


def start_point(obs: list[StationEstimate]) -> tuple[float, float]:
    """Mean of the single-station epicentres (each station's distance along
    its back-azimuth), or the station centroid when no station has one."""
    pts = [destination(o.lat, o.lon, o.baz_rad, float(np.exp(o.log_dist)))
           for o in obs if np.isfinite(o.baz_rad) and o.kappa > 0]
    if not pts:
        pts = [(o.lat, o.lon) for o in obs]
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def locate(obs: list[StationEstimate], vp: float = 6.0, depth_km: float = 10.0,
           sigma_p: float = 0.5, half_deg: float = 3.0, step_deg: float = 0.05) -> Location | None:
    """Grid-search epicentre. None when the stations cannot fix one: fewer
    than two stations and no back-azimuth."""
    if not obs or (len(obs) < 2 and not any(np.isfinite(o.baz_rad) and o.kappa > 0 for o in obs)):
        return None
    lat0, lon0 = start_point(obs)

    def grid(clat, clon, half, step):
        n = int(np.floor(2 * half / step + 1e-9)) + 1
        a = np.arange(n) * step - half
        return np.meshgrid(clat + a, clon + a, indexing="ij")

    la, lo = grid(lat0, lon0, half_deg, step_deg)
    j_coarse = terms(la, lo, obs, vp, depth_km, sigma_p)[0].sum(axis=0)
    k = np.unravel_index(np.argmin(j_coarse), j_coarse.shape)
    fla, flo = grid(la[k], lo[k], 0.1, 0.005)
    j_fine = terms(fla, flo, obs, vp, depth_km, sigma_p)[0].sum(axis=0)
    kf = np.unravel_index(np.argmin(j_fine), j_fine.shape)
    best_lat, best_lon, best_j = float(fla[kf]), float(flo[kf]), float(j_fine[kf])
    if j_coarse[k] < best_j:           # the fine grid contains the coarse point; only rounding
        best_lat, best_lon, best_j = float(la[k]), float(lo[k]), float(j_coarse[k])

    _, origin, resid = terms(best_lat, best_lon, obs, vp, depth_km, sigma_p)
    inside = j_coarse <= best_j + ONE_SIGMA_DJ
    err = distance_km(best_lat, best_lon, la[inside], lo[inside])
    err_km = float(max(err.max() if err.size else 0.0,
                       distance_km(0.0, 0.0, 0.0, step_deg) / 2))
    return Location(best_lat, best_lon, float(origin), float(np.sqrt(np.mean(resid ** 2))),
                    best_j, err_km, len(obs))


def locate_robust(obs: list[StationEstimate], max_rms: float = 2.0, **kw) -> Location | None:
    """`locate`, then while the P rms exceeds `max_rms` and more than two
    stations remain, drops the station contributing most to J and relocates.
    ayzek's `Network::locate` does the same."""
    obs, dropped = list(obs), []
    while True:
        loc = locate(obs, **kw)
        if loc is None or loc.rms <= max_rms or len(obs) <= 2:
            if loc is not None:
                loc.dropped = dropped
            return loc
        parts = terms(loc.lat, loc.lon, obs, kw.get("vp", 6.0), kw.get("depth_km", 10.0),
                      kw.get("sigma_p", 0.5))[0]
        worst = int(np.argmax(parts))
        dropped.append(obs[worst].station)
        del obs[worst]
