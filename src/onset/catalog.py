"""The AFAD catalogue, station coordinates and TauP arrivals. Needs obspy.

Besides labelling each event trace's P, the catalogue answers the question
that keeps the training set honest: *did some other catalogued event arrive at
this station inside this window?* An event trace whose window holds an earlier
event's arrival before its own P has signal under a "noise" label. A noise
window holding any arrival is not noise, and neither is one that falls in the
coda of an event that arrived shortly before it (`Catalog.ringing`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from obspy.taup import TauPyModel

P_PHASES = ("p", "P", "Pn", "Pg")
S_PHASES = ("s", "S", "Sn", "Sg")

# (max distance km, min magnitude): a catalogued event counts as visible at a
# station if any rule admits it. Deliberately generous: dropping a clean
# window costs a little data, keeping a contaminated one teaches the model
# that earthquakes are noise.
VISIBILITY = ((50.0, 0.0), (150.0, 2.0), (400.0, 3.0), (1500.0, 4.5))
MAX_TRAVEL_S = 200.0
CODA_MAX_S = 3600.0


def coda_seconds(magnitude: float, cap: float = CODA_MAX_S) -> float:
    """How long an event's signal lasts after its P, from the duration
    magnitude relation Md = 2 log10(tau) - 0.87 inverted: about 25 s at M2,
    85 s at M3, 270 s at M4 and 860 s at M5, capped at `cap` (the relation
    is for local events; a great earthquake's aftershocks are catalogued
    events of their own)."""
    return float(min(cap, 10.0 ** ((magnitude + 0.87) / 2.0)))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


class TravelTimes:
    """iasp91 first arrivals on a 1 km (distance, depth) grid, cached."""

    def __init__(self, grid_km: float = 1.0):
        self.model = TauPyModel("iasp91")
        self.grid = grid_km
        self.cache: dict = {}

    def first(self, dist_km: float, depth_km: float, phases=P_PHASES) -> float | None:
        key = (round(dist_km / self.grid), round(max(depth_km, 0.0) / self.grid), phases)
        if key not in self.cache:
            arr = self.model.get_travel_times(
                source_depth_in_km=key[1] * self.grid,
                distance_in_degree=key[0] * self.grid / 111.195,
                phase_list=list(phases))
            self.cache[key] = arr[0].time if arr else None
        return self.cache[key]


@dataclass
class Event:
    event_id: int
    origin: float          # POSIX seconds, UTC
    lat: float
    lon: float
    depth_km: float
    magnitude: float


class Catalog:
    def __init__(self, path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        t = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M:%S", utc=True, errors="coerce")
        # Seconds via a timedelta, not astype(int64): pandas may parse to s, ms
        # or ns resolution, and the integer's unit follows it.
        epoch = pd.Timestamp(0, tz="UTC")
        df = df.assign(t=(t - epoch).dt.total_seconds()).dropna(subset=["t"])
        df = df.sort_values("t").reset_index(drop=True)
        self.t = df["t"].to_numpy()
        self.lat = df["Latitude"].to_numpy(float)
        self.lon = df["Longitude"].to_numpy(float)
        self.depth = df["Depth"].to_numpy(float)
        self.mag = df["Magnitude"].to_numpy(float)
        self.ids = df["EventID"].to_numpy()
        self.row = {int(e): i for i, e in enumerate(self.ids)}
        self.max_mag = float(self.mag.max()) if len(self.mag) else 0.0

    def event(self, event_id: int) -> Event | None:
        i = self.row.get(int(event_id))
        if i is None:
            return None
        return Event(int(self.ids[i]), float(self.t[i]), float(self.lat[i]),
                     float(self.lon[i]), float(self.depth[i]), float(self.mag[i]))

    def arrivals(self, lat: float, lon: float, t0: float, t1: float, taup: TravelTimes,
                 exclude: int | None = None) -> list[tuple[int, float]]:
        """Visible catalogued P arrivals at (lat, lon) inside [t0, t1].

        Returns:
            [(event_id, arrival time)], earliest first.
        """
        a, b = np.searchsorted(self.t, [t0 - MAX_TRAVEL_S, t1])
        if a == b:
            return []
        idx = np.arange(a, b)
        d = haversine_km(lat, lon, self.lat[idx], self.lon[idx])
        seen = np.zeros(len(idx), bool)
        for dmax, mmin in VISIBILITY:
            seen |= (d <= dmax) & (self.mag[idx] >= mmin)
        out = []
        for i, dist in zip(idx[seen], d[seen]):
            if exclude is not None and int(self.ids[i]) == exclude:
                continue
            tt = taup.first(float(dist), float(self.depth[i]))
            if tt is not None and t0 <= self.t[i] + tt <= t1:
                out.append((int(self.ids[i]), float(self.t[i] + tt)))
        return sorted(out, key=lambda e: e[1])


    def ringing(self, lat: float, lon: float, t0: float, t1: float, taup: TravelTimes,
                coda_cap: float = CODA_MAX_S) -> list[tuple[int, float]]:
        """Visible catalogued events whose signal overlaps [t0, t1] at (lat, lon):
        their P arrives by t1 and their coda (`coda_seconds`) has not ended by
        t0. With `coda_cap` 0 this is `arrivals`: a P inside the window only.

        Returns:
            [(event_id, P arrival time)], earliest first.
        """
        look = MAX_TRAVEL_S + (coda_seconds(self.max_mag, coda_cap) if coda_cap > 0 else 0.0)
        a, b = np.searchsorted(self.t, [t0 - look, t1])
        if a == b:
            return []
        idx = np.arange(a, b)
        d = haversine_km(lat, lon, self.lat[idx], self.lon[idx])
        seen = np.zeros(len(idx), bool)
        for dmax, mmin in VISIBILITY:
            seen |= (d <= dmax) & (self.mag[idx] >= mmin)
        out = []
        for i, dist in zip(idx[seen], d[seen]):
            tt = taup.first(float(dist), float(self.depth[i]))
            if tt is None:
                continue
            p = self.t[i] + tt
            end = p + (coda_seconds(self.mag[i], coda_cap) if coda_cap > 0 else 0.0)
            if p <= t1 and end >= t0:
                out.append((int(self.ids[i]), float(p)))
        return sorted(out, key=lambda e: e[1])


def load_stations(path) -> dict:
    """{(network, station): (lat, lon)}, plus {station: (lat, lon)} where the
    code is unique across networks."""
    df = pd.read_csv(path)
    out = {(r.network, r.station): (float(r.latitude), float(r.longitude))
           for r in df.itertuples()}
    counts = df["station"].value_counts()
    for r in df.itertuples():
        if counts[r.station] == 1:
            out[r.station] = (float(r.latitude), float(r.longitude))
    return out


def distance_km(lat1, lon1, lat2, lon2) -> float:
    return float(haversine_km(lat1, lon1, lat2, lon2))


def seconds_to_samples(t: float, fs: float) -> float:
    return t * fs if t is not None and math.isfinite(t) else float("nan")
