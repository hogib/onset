"""`onset build-fdsn`: the FDSN catalogue pulls -> an onset dataset.

Reads three sibling pulls of the same events (see docs/DATA.md):

    events   window_post_60s/event_<id>_raw.mseed          origin .. origin+60 s
    context  noise_pre_3h/noise_event_<id>_raw.mseed       5 min, 3 h before origin
    noise    noise_pre_6h/noise_event_<id>_raw.mseed       5 min, 6 h before origin

A file can hold several stations; each station becomes its own trace. For
each station:

1. Choose one 3-component instrument (HH over BH over EH; blank location first).
2. Put it on a 100 Hz grid with gaps masked, and filter it causally (`dsp`).
3. Events: predict P and S with TauP, refine P with an AIC pick
   (`labels.refine_p`), and drop the trace if another catalogued event
   arrives before its P.
4. Context and noise: keep the last `--context-seconds` / `--noise-seconds`
   (the rest warms up the filter), and drop them if a visible catalogued
   event arrives inside, or arrived early enough before that its coda is
   still ringing (`catalog.coda_seconds`, capped at `--coda-max-s`).
5. Noise only (not context, which describes the station's own background,
   busy or not): drop windows the catalogue cannot vouch for, because small
   events it does not list are likely there. That is a busy time and place
   (more than `--max-active-events` catalogued events within
   `--active-radius-km` in the `--active-hours` around it), or the aftermath
   of a large event (`catalog.aftermath_of`), and any window
   `onset audit-noise` found in a multi-station coincidence
   (`--exclude-noise`).

Every drop is counted by reason in build.json.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime, read
from obspy.geodetics import gps2dist_azimuth

from onset.catalog import (S_PHASES, VISIBILITY, Catalog, TravelTimes,
                           distance_km, load_stations)
from onset.config import SAMPLE_RATE
from onset.dsp import filter_components
from onset.labels import refine_p
from onset.store import StoreWriter, station_split

BANDS = ("HH", "BH", "EH")
COMPONENT_SETS = (("Z", "N", "E"), ("Z", "1", "2"))
TOLERANCE_S = {"aic": 0.3, "taup": 1.0}

_W: dict = {}                     # per-worker state, filled by _init


def _init(catalog_path, stations_path, cfg):
    _W["catalog"] = Catalog(catalog_path)
    _W["stations"] = load_stations(stations_path)
    _W["taup"] = TravelTimes()
    _W["cfg"] = cfg


# ---------------------------------------------------------------------------
# miniSEED -> arrays
# ---------------------------------------------------------------------------

def instruments(st: Stream) -> dict:
    """{(network, station): [Z, N, E] trace lists} for the preferred instrument."""
    return {k: comps for k, (comps, _) in instruments_with_set(st).items()}


def instruments_with_set(st: Stream) -> dict:
    """{(network, station): ([Z, N, E] trace lists, "ZNE" or "Z12")}. A Z12
    instrument's horizontals are not known to point north and east, so its
    back-azimuth label is withheld."""
    groups = defaultdict(list)
    for tr in st:
        ch = tr.stats.channel
        if len(ch) == 3 and ch[:2] in BANDS:
            groups[(tr.stats.network, tr.stats.station, tr.stats.location, ch[:2])].append(tr)
    best = {}
    for (net, sta, loc, band), trs in groups.items():
        comps = {tr.stats.channel[2] for tr in trs}
        chosen = next((cs for cs in COMPONENT_SETS if set(cs) <= comps), None)
        if chosen is None:
            continue
        rank = (BANDS.index(band), loc != "", loc)
        if (net, sta) not in best or rank < best[(net, sta)][0]:
            best[(net, sta)] = (rank, ([[t for t in trs if t.stats.channel[2] == c]
                                        for c in chosen], "".join(chosen)))
    return {k: v[1] for k, v in best.items()}


def to_grid(traces, t0: UTCDateTime, n: int, fs: float = SAMPLE_RATE):
    """One component's traces -> (n,) float64 samples and (n,) missing mask."""
    st = Stream([tr.copy() for tr in traces])
    for tr in st:
        if abs(tr.stats.sampling_rate - fs) > 1e-6:
            tr.resample(fs)
    st.merge(method=0, fill_value=None)
    tr = st[0]
    tr.trim(t0, t0 + (n - 1) / fs, pad=True, nearest_sample=True, fill_value=None)
    data = np.ma.masked_invalid(np.ma.asarray(tr.data, dtype=np.float64))
    x = np.zeros(n)
    missing = np.ones(n, bool)
    m = min(n, len(data))
    x[:m] = data.filled(0.0)[:m]
    missing[:m] = np.ma.getmaskarray(data)[:m]
    return x, missing


def load_window(components, t0: UTCDateTime, n: int):
    """[Z, N, E] trace lists -> filtered (n, 3) float32 and combined missing (n,)."""
    x = np.zeros((n, 3))
    missing = np.ones((n, 3), bool)
    for c, traces in enumerate(components):
        x[:, c], missing[:, c] = to_grid(traces, t0, n)
    return filter_components(x, missing)


def load_tail(components, seconds: float):
    """The last `seconds` of a file, filtered from the file's start so the
    filter has warmed up. Returns (wave, missing, start time) or None."""
    t_start = min(tr.stats.starttime for comp in components for tr in comp)
    t_end = max(tr.stats.endtime for comp in components for tr in comp)
    n_full = int(round((t_end - t_start) * SAMPLE_RATE)) + 1
    n = int(seconds * SAMPLE_RATE)
    if n_full < n:
        return None
    wave, missing = load_window(components, t_start, n_full)
    return wave[-n:], missing[-n:], t_start + (n_full - n) / SAMPLE_RATE


# ---------------------------------------------------------------------------
# One file
# ---------------------------------------------------------------------------

def _clean_tail(components, seconds, lat, lon, max_missing, drops, what):
    got = load_tail(components, seconds)
    if got is None:
        drops[f"{what}_short"] += 1
        return None
    wave, missing, t0 = got
    if missing.mean() > max_missing:
        drops[f"{what}_gappy"] += 1
        return None
    t1 = float(t0.timestamp) + seconds
    if _W["catalog"].ringing(lat, lon, float(t0.timestamp), t1, _W["taup"],
                             _W["cfg"]["coda_max_s"]):
        drops[f"{what}_contaminated"] += 1
        return None
    return wave, missing, t0


def _noise_vouched(event_id, net, sta, lat, lon, t0, seconds, drops) -> bool:
    """False, counting the reason, for a noise window that is likely to hold
    events the catalogue does not list."""
    cfg, cat = _W["cfg"], _W["catalog"]
    key = f"noise/{event_id}/{net}.{sta}"
    if key in cfg["exclude_keys"] or event_id in cfg["exclude_pulls"]:
        drops["noise_excluded"] += 1
        return False
    if cfg["max_active_events"] >= 0 and cfg["active_hours"] > 0:
        h = cfg["active_hours"] * 3600.0
        if cat.nearby_count(lat, lon, t0 - h, t0 + seconds + h,
                            cfg["active_radius_km"]) > cfg["max_active_events"]:
            drops["noise_active"] += 1
            return False
    if cfg["aftermath_days"] > 0 and cat.aftermath_of(
            lat, lon, t0, cfg["aftermath_radius_km"], cfg["aftermath_mag"],
            cfg["aftermath_days"]):
        drops["noise_aftermath"] += 1
        return False
    return True


def load_exclusions(paths, min_stations: int = 3):
    """From `onset audit-noise` CSVs: the noise windows in a multi-station
    coincidence, and the pulls where one spans `min_stations` or more (a
    clear event, likely below the trigger at the other stations too)."""
    import pandas as pd
    keys, pulls = set(), set()
    for p in paths:
        t = pd.read_csv(p)
        t = t[t["coincident"].astype(bool)]
        keys |= set(t["key"])
        n = t.groupby("pull")["station"].nunique()
        pulls |= {int(x) for x in n[n >= min_stations].index}
    return keys, pulls


def process_file(path: Path):
    """Returns (records, drops): records are (key, wave, missing, meta)."""
    cfg, cat, taup = _W["cfg"], _W["catalog"], _W["taup"]
    drops, records = Counter(), []
    m = re.search(r"event_(\d+)_raw", path.name)
    ev = cat.event(int(m.group(1))) if m else None
    if ev is None:
        drops["no_catalog_entry"] += 1
        return records, drops
    try:
        st = read(str(path))
    except Exception:
        drops["unreadable"] += 1
        return records, drops

    side = {}
    for what, d in (("context", cfg["context_dir"]), ("noise", cfg["noise_dir"])):
        p = Path(d) / f"noise_{path.name}" if d else None
        try:
            side[what] = instruments(read(str(p))) if p and p.exists() else {}
        except Exception:
            side[what] = {}

    fs = SAMPLE_RATE
    n = int(cfg["event_seconds"] * fs)
    origin = UTCDateTime(ev.origin)
    offset = cfg["event_start_offset"]            # window start relative to origin
    t0 = origin + offset
    for (net, sta), (comps, comp_set) in instruments_with_set(st).items():
        coords = _W["stations"].get((net, sta)) or _W["stations"].get(sta)
        if coords is None:
            drops["no_station_coords"] += 1
            continue
        lat, lon = coords
        dist = distance_km(ev.lat, ev.lon, lat, lon)
        # A far station for a small event records nothing: labelling its
        # window "event" would teach the model to fire on noise.
        if not any(dist <= dmax and ev.magnitude >= mmin for dmax, mmin in VISIBILITY):
            drops["not_visible_at_distance"] += 1
            continue
        wave, missing = load_window(comps, t0, n)
        if missing.mean() > cfg["max_missing"]:
            drops["event_gappy"] += 1
            continue
        tp = taup.first(dist, ev.depth_km)
        ts = taup.first(dist, ev.depth_km, S_PHASES)
        if tp is None or (tp - offset) * fs > n - 2 * fs:
            drops["p_outside_window"] += 1
            continue
        if cat.arrivals(lat, lon, float(t0.timestamp) - 30.0, ev.origin + tp, taup,
                        exclude=ev.event_id):
            drops["earlier_arrival_in_window"] += 1
            continue
        p_pred = (tp - offset) * fs
        s_pred = (ts - offset) * fs if ts is not None else float("nan")
        pick, ok, snr = refine_p(wave[:, 0], missing, p_pred, fs, s_pred)
        source = "aic" if ok else "taup"

        split = station_split(net, sta)
        baz = gps2dist_azimuth(lat, lon, ev.lat, ev.lon)[1]    # station -> event
        base = {"split": split, "source": "fdsn", "network": net, "station": sta,
                "event_id": ev.event_id, "magnitude": ev.magnitude,
                "depth_km": ev.depth_km, "distance_km": dist,
                "station_lat": lat, "station_lon": lon,
                "event_lat": ev.lat, "event_lon": ev.lon, "components": comp_set,
                "back_azimuth_deg": baz if comp_set == "ZNE" else float("nan")}
        ctx_key = None
        if (net, sta) in side["context"]:
            got = _clean_tail(side["context"][(net, sta)], cfg["context_seconds"],
                              lat, lon, cfg["max_context_missing"], drops, "context")
            if got is not None:
                ctx_key = f"context/{ev.event_id}/{net}.{sta}"
                records.append((ctx_key, got[0], got[1],
                                {**base, "kind": "context", "start_time": str(got[2])}))
        else:
            drops["context_absent"] += 1

        records.append((f"event/{ev.event_id}/{net}.{sta}", wave, missing, {
            **base, "kind": "event", "start_time": str(t0), "p_sample": pick,
            "p_source": source, "p_tolerance_s": TOLERANCE_S[source],
            "p_predicted_sample": p_pred, "s_sample": s_pred, "pick_snr": snr,
            "context_key": ctx_key}))

        if (net, sta) in side["noise"]:
            got = _clean_tail(side["noise"][(net, sta)], cfg["noise_seconds"],
                              lat, lon, cfg["max_missing"], drops, "noise")
            if got is not None and not _noise_vouched(ev.event_id, net, sta, lat, lon,
                                                      float(got[2].timestamp),
                                                      cfg["noise_seconds"], drops):
                got = None
            if got is not None:
                records.append((f"noise/{ev.event_id}/{net}.{sta}", got[0], got[1],
                                {**base, "kind": "noise", "start_time": str(got[2]),
                                 "context_key": ctx_key}))
        else:
            drops["noise_absent"] += 1
    return records, drops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# The data_downloader checkout the pulls live in; every path below is relative
# to it unless given explicitly.
FDSN_ROOT = os.environ.get("ONSET_FDSN_ROOT", "/home/oguzb/Projects/sismokaos/data_downloader")


def parse_args(argv=None):
    root = FDSN_ROOT
    p = argparse.ArgumentParser(prog="onset build-fdsn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events-dir", default=f"{root}/raw/data/batched_waveforms/window_post_60s")
    p.add_argument("--context-dir", default=f"{root}/raw/data/batched_noise_waveforms/noise_pre_3h")
    p.add_argument("--noise-dir", default=f"{root}/raw/data/batched_noise_waveforms/noise_pre_6h")
    p.add_argument("--catalog", default=f"{root}/catalogs/catalog_current.csv")
    p.add_argument("--stations", default=f"{root}/catalogs/station_coords.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--event-seconds", type=float, default=60.0)
    p.add_argument("--event-start-offset", type=float, default=0.0,
                   help="Event window start relative to origin, seconds: 0 for "
                        "window_post_60s, -60 for a pull that starts a minute early.")
    p.add_argument("--context-seconds", type=float, default=120.0)
    p.add_argument("--noise-seconds", type=float, default=120.0)
    p.add_argument("--max-missing", type=float, default=0.10,
                   help="Largest missing fraction for an event or noise trace.")
    p.add_argument("--max-context-missing", type=float, default=0.05)
    p.add_argument("--max-active-events", type=int, default=3,
                   help="Drop a noise window when more catalogued events than this lie "
                        "within --active-radius-km in the --active-hours around it; "
                        "-1 turns the rule off.")
    p.add_argument("--active-radius-km", type=float, default=75.0)
    p.add_argument("--active-hours", type=float, default=12.0)
    p.add_argument("--aftermath-mag", type=float, default=5.0,
                   help="Drop noise windows in the aftermath of an event this large or "
                        "larger within --aftermath-radius-km ...")
    p.add_argument("--aftermath-radius-km", type=float, default=150.0)
    p.add_argument("--aftermath-days", type=float, default=30.0,
                   help="... for this many days at --aftermath-mag, x3.2 per magnitude "
                        "unit above it; 0 turns the rule off.")
    p.add_argument("--exclude-noise", nargs="*", default=[],
                   help="onset audit-noise CSVs: their coincident noise windows are dropped.")
    p.add_argument("--coda-max-s", type=float, default=3600.0,
                   help="Longest coda a noise or context window is checked back "
                        "for; 0 checks only for arrivals inside the window.")
    p.add_argument("--limit", type=int, default=None, help="First N event files only.")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    files = sorted(Path(a.events_dir).glob("event_*_raw.mseed"))[: a.limit]
    cfg = {k: getattr(a, k) for k in ("events_dir", "context_dir", "noise_dir",
                                       "event_seconds", "event_start_offset",
                                       "context_seconds",
                                       "noise_seconds", "max_missing",
                                       "max_context_missing", "coda_max_s",
                                       "max_active_events", "active_radius_km",
                                       "active_hours", "aftermath_mag",
                                       "aftermath_radius_km", "aftermath_days")}
    cfg["exclude_keys"], cfg["exclude_pulls"] = load_exclusions(a.exclude_noise)
    if a.exclude_noise:
        print(f"  excluding {len(cfg['exclude_keys'])} noise windows and "
              f"{len(cfg['exclude_pulls'])} whole pulls from {len(a.exclude_noise)} audit(s)")
    writer = StoreWriter(a.out)
    drops, kinds = Counter(), Counter()
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(a.catalog, a.stations, cfg)) as pool:
        for i, (records, d) in enumerate(pool.map(process_file, files, chunksize=16), 1):
            drops.update(d)
            for key, wave, missing, meta in records:
                writer.add(key, wave, missing, meta)
                kinds[meta["kind"]] += 1
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files  {dict(kinds)}  "
                      f"{time.time() - t0:.0f}s", flush=True)
    writer.close({"args": vars(a), "files": len(files), "written": dict(kinds),
                  "dropped": dict(drops.most_common())})
    print(f"wrote {dict(kinds)} to {a.out}")
    print(f"dropped: {dict(drops.most_common())}")


if __name__ == "__main__":
    main()
