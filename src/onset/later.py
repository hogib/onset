"""Catalogued arrivals after an event trace's own P: onsets in its coda.

An event window runs 60 s or more past origin, and in a sequence it often
holds another catalogued event after its own P. Left unlabelled, that onset
is trained as more coda: dt keeps counting through it, which is the opposite
of the restart ayzek's trigger fires on. On fdsn_wide_x, 5.6% of event traces
hold one, 1.7% within reach of a 40 s crop.

Each one is predicted with TauP and refined by AIC like the first P
(`labels.refine_p`), stopping 0.5 s short of its own predicted S:

- `a`, an accepted pick: a labelled onset, dt restarts there, with a
  tolerance of `LATER_TOL_S` (wider than the first P's 0.3 s: the coda
  before it is louder than the noise before a first P, and AIC weaker);
- `m`, no acceptable pick (too small for the coda, a gap, or within a second
  of the first event's S, which AIC cannot tell from it): only the TauP
  prediction, around which dt is not trained (`labels.token_targets`).

They are stored in index.csv's `later_p`, as `sample:tolerance_s:kind`
entries joined by `;`, in samples of the stored trace. `onset label-later`
adds the column to an existing store; `build-fdsn` writes it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

LATER_TOL_S = 0.5
TAUP_TOL_S = 1.0


def later_onsets(cat, taup, lat: float, lon: float, event_id: int, t0: float, fs: float,
                 z: np.ndarray, missing: np.ndarray, p_sample: float,
                 s_sample: float = float("nan")) -> list[tuple[float, float, str]]:
    """[(sample, tolerance s, "a" or "m")] for the visible catalogued arrivals
    after `p_sample` (+ 0.5 s) in a trace starting at POSIX time `t0`."""
    from onset.catalog import S_PHASES, distance_km
    from onset.labels import refine_p

    out = []
    end = t0 + len(z) / fs
    for eid, t_arr in cat.arrivals(lat, lon, t0 + p_sample / fs + 0.5, end, taup,
                                   exclude=int(event_id)):
        ev = cat.event(eid)
        guess = (t_arr - t0) * fs
        ts = taup.first(distance_km(ev.lat, ev.lon, lat, lon), ev.depth_km, S_PHASES)
        s_guess = (ev.origin + ts - t0) * fs if ts is not None else None
        pick, ok, _ = refine_p(z, missing, guess, fs, s_guess)
        near_s = np.isfinite(s_sample) and abs(pick - s_sample) < fs
        if ok and pick > p_sample + fs and not near_s:
            out.append((float(pick), LATER_TOL_S, "a"))
        else:
            out.append((float(guess), TAUP_TOL_S, "m"))
    return out


def encode(onsets) -> str | None:
    return ";".join(f"{s:.1f}:{tol:g}:{k}" for s, tol, k in onsets) or None


def decode(value) -> list[tuple[float, float, str]]:
    if not isinstance(value, str) or not value:
        return []
    out = []
    for part in value.split(";"):
        s, tol, k = part.split(":")
        out.append((float(s), float(tol), k))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="onset label-later", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset store to label in place")
    ap.add_argument("--catalog", help="default: the store's build.json")
    ap.add_argument("--stations", help="default: the store's build.json")
    a = ap.parse_args(argv)

    from obspy import UTCDateTime

    from onset.catalog import Catalog, TravelTimes, load_stations
    from onset.store import INDEX, StoreReader

    root = Path(a.data)
    args = json.loads((root / "build.json").read_text()).get("args", {})
    cat = Catalog(a.catalog or args["catalog"])
    stations = load_stations(a.stations or args["stations"])
    taup = TravelTimes()
    store = StoreReader(root)
    idx = store.index
    fs = 100.0
    labels = pd.Series([None] * len(idx), dtype=object)
    counts = {"a": 0, "m": 0}
    traces = 0
    for i, r in idx.iterrows():
        if r.kind != "event" or r.source != "fdsn" or not np.isfinite(r.p_sample):
            continue
        c = stations.get((r.network, r.station)) or stations.get(r.station)
        if c is None:
            continue
        t0 = UTCDateTime(r.start_time).timestamp
        # Cheap catalogue test first; the waveform is read only for a hit.
        if not cat.arrivals(c[0], c[1], t0 + r.p_sample / fs + 0.5,
                            t0 + r.n_samples / fs, taup, exclude=int(r.event_id)):
            continue
        wave, missing = store.read(r.key)
        found = later_onsets(cat, taup, c[0], c[1], r.event_id, t0, fs, wave[:, 0], missing,
                             float(r.p_sample),
                             float(r.s_sample) if pd.notna(r.s_sample) else float("nan"))
        labels[i] = encode(found)
        traces += 1
        for _, _, k in found:
            counts[k] += 1
    backup = root / (INDEX + ".bak")
    if not backup.exists():
        shutil.copy(root / INDEX, backup)
    idx["later_p"] = labels
    idx.to_csv(root / INDEX, index=False)
    print(f"{traces} event traces with later catalogued arrivals: {counts['a']} picked "
          f"(labelled onsets), {counts['m']} not (dt masked around TauP); "
          f"index backed up to {backup.name}")
