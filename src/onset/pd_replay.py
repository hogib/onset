"""Pd magnitudes for the catalogued events in a continuous recording.

    onset pd-replay day.mseed [more.mseed ...] --fit runs/pd_v4/fit.json

For every catalogued event of at least --min-mag in the recording's span,
each station's P is predicted with TauP and refined by AIC (labels.refine_p),
Pd is measured as in pd.py, and the event's magnitude is estimated from the
stations present as in pd_fit.estimate_censored, for every window of the fit.
A value whose window contains a sample within CLIP_FRACTION of the 24-bit
digitiser's full scale is flagged: a clipped trace understates Pd. The
check is meant for the large events that the stored training windows lack,
such as the 2023-02-06 Kahramanmaras sequence at GAZ and KMRS.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from onset.pd import FDSN_ROOT, FS, NOISE_S, SETTLE_S, PD_WINDOWS_S, displacement, peak_displacement

FULL_SCALE = 2 ** 23
CLIP_FRACTION = 0.95


def main(argv=None):
    ap = argparse.ArgumentParser(prog="onset pd-replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="miniSEED of one continuous recording")
    ap.add_argument("--fit", required=True, help="fit.json from onset fit-pd")
    ap.add_argument("--catalog", default=f"{FDSN_ROOT}/catalogs/catalog_current.csv")
    ap.add_argument("--stations", default=f"{FDSN_ROOT}/catalogs/station_coords.csv")
    ap.add_argument("--inventory", default=f"{FDSN_ROOT}/raw/data/station_inventory")
    ap.add_argument("--min-mag", type=float, default=4.5)
    ap.add_argument("--max-km", type=float, default=250.0)
    a = ap.parse_args(argv)

    from obspy import Stream, UTCDateTime, read
    from onset import pd as pdm
    from onset.build_fdsn import instruments, to_grid
    from onset.catalog import P_PHASES, S_PHASES, Catalog, TravelTimes, distance_km, load_stations
    from onset.dsp import causal_filter
    from onset.labels import refine_p
    from onset.pd_fit import estimate_censored

    fit = json.loads(open(a.fit).read())
    cat, coords, taup = Catalog(a.catalog), load_stations(a.stations), TravelTimes()
    types = pd.read_csv(a.catalog, encoding="utf-8-sig").set_index("EventID").Type.to_dict()
    pdm._init(a.inventory)
    st = Stream()
    for f in a.files:
        st += read(f)
    t0 = min(tr.stats.starttime for tr in st)
    n = int((max(tr.stats.endtime for tr in st) - t0) * FS) + 1
    traces = {}
    for (net, sta), comps in instruments(st).items():
        counts, missing = to_grid(comps[0], t0, n)
        traces[(net, sta)] = (counts, missing, causal_filter(counts, missing), comps[0][0].stats)

    lo, hi = np.searchsorted(cat.t, [t0.timestamp, t0.timestamp + n / FS])
    rows = []
    for i in range(lo, hi):
        if cat.mag[i] < a.min_mag:
            continue
        eid = int(cat.ids[i])
        for (net, sta), (counts, missing, z, stats) in traces.items():
            c = coords.get((net, sta)) or coords.get(sta)
            if c is None:
                continue
            dist = distance_km(cat.lat[i], cat.lon[i], *c)
            tp = taup.first(dist, cat.depth[i], P_PHASES)
            ts = taup.first(dist, cat.depth[i], S_PHASES)
            if dist > a.max_km or tp is None:
                continue
            guess = (cat.t[i] + tp - t0.timestamp) * FS
            s_guess = (cat.t[i] + ts - t0.timestamp) * FS if ts is not None else None
            pick, ok, _ = refine_p(z, missing, guess, FS, s_guess)
            p = int(round(pick))
            s0, s1 = p - int((SETTLE_S + NOISE_S + 1.0) * FS), p + int(max(PD_WINDOWS_S) * FS)
            if s0 < 0 or s1 > n or missing[s0:s1].any():
                continue
            resp = pdm._response(stats.network, stats.station, stats.location, stats.channel,
                                 UTCDateTime(t0.timestamp + p / FS))
            if resp is None:
                continue
            d = displacement(counts[s0:s1], resp[0], resp[1])
            row = {"event_id": eid, "magnitude": float(cat.mag[i]), "type": types.get(eid, ""),
                   "origin": str(UTCDateTime(cat.t[i]))[:19], "station": sta,
                   "distance_km": dist, "aic": ok, **peak_displacement(d, p - s0)}
            for t in PD_WINDOWS_S:
                w = counts[p: p + int(t * FS)]
                row[f"clip_{t:g}s"] = bool(np.abs(w - np.median(counts[s0:p])).max()
                                          >= CLIP_FRACTION * FULL_SCALE)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no catalogued event with a usable P window")
        return
    print(f"{df.event_id.nunique()} events of M{a.min_mag:g} or more, {len(df)} station values\n")
    for eid, g in df.groupby("event_id", sort=False):
        r = g.iloc[0]
        print(f"{r.origin}  {r.type} {r.magnitude:.1f}  (event {eid})")
        for x in g.itertuples():
            print(f"    {x.station:5s} {x.distance_km:5.0f} km  P {'AIC' if x.aic else 'TauP'}  "
                  + "  ".join(f"Pd{t:g} {getattr(x, f'pd_{t:g}s'):.1e}"
                              + ("*" if getattr(x, f"clip_{t:g}s") else "")
                              for t in (1, 3, 5, 10))
                  + f"  noise {x.pd_noise:.1e}")
        est = []
        for t in PD_WINDOWS_S:
            f = fit["windows"][f"{t:g}"]
            e = estimate_censored(g.assign(split="replay"), f)
            est.append(f"{t:g} s: " + (f"M{e.est.iloc[0]:.1f}±{e.sd.iloc[0]:.1f}" if len(e) else "-"))
        print("    estimate  " + "   ".join(est))
    print("\n* a sample of the window within 5% of the digitiser's full scale (clipped)")
