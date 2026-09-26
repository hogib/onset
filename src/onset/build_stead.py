"""STEAD: a fallback training source, and the check on our FDSN labels.

STEAD (Mousavi et al. 2019) has what the FDSN pulls lack, manual P and S
picks, and lacks what they have: Turkish stations, station context, and more
than about 10 s before P. So it is used two ways:

- `onset build-stead` writes STEAD traces in the store format (`store.py`),
  with `p_source="manual"`. Training can mix them in with `--fallback`.
- `onset validate-aic` measures the label pipeline `build-fdsn` relies on. It
  hides each manual pick behind a TauP-like prediction error, runs the same
  AIC refinement, and reports how far the refined pick lands from the truth.
  That is the only ground truth available for the FDSN labels'
  `p_tolerance_s`.

STEAD's waveforms are stored E, N, Z and are written here Z, N, E, filtered with
the same causal band-pass as everything else. Noise is STEAD's chunk1, which is
not on disk; `trace_category` is checked regardless.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from onset.config import SAMPLE_RATE
from onset.dsp import filter_components
from onset.labels import refine_p
from onset.store import StoreWriter, station_split

ENZ_TO_ZNE = [2, 1, 0]
MANUAL_TOLERANCE_S = 0.1
STEAD_DIR = os.environ.get("ONSET_STEAD_DIR", "/home/oguzb/Projects/Codings/STEAD")


def chunks(stead_dir) -> list[tuple[Path, Path]]:
    """(csv, hdf5) pairs that are both present."""
    out = []
    for csv in sorted(Path(stead_dir).glob("chunk*.csv")):
        h5 = csv.with_suffix(".hdf5")
        if h5.exists():
            out.append((csv, h5))
    return out


def load_metadata(stead_dir, manual_only=True) -> pd.DataFrame:
    frames = []
    for csv, h5 in chunks(stead_dir):
        df = pd.read_csv(csv, low_memory=False)
        frames.append(df.assign(_h5=str(h5)))
    if not frames:
        raise FileNotFoundError(f"no chunk*.csv with a matching .hdf5 under {stead_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.trace_category == "earthquake_local"]
    if manual_only:
        df = df[df.p_status == "manual"]
    return df


def read_trace(h5: h5py.File, name: str):
    """STEAD trace -> filtered (6000, 3) Z N E and missing mask, or None for a
    trace with a dead component."""
    x = h5[f"data/{name}"][()][:, ENZ_TO_ZNE].astype(np.float64)
    if (np.abs(x).max(axis=0) == 0).any():
        return None
    return filter_components(x, np.zeros(x.shape, bool), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# build-stead
# ---------------------------------------------------------------------------

def build(argv=None):
    p = argparse.ArgumentParser(prog="onset build-stead")
    p.add_argument("--stead-dir", default=STEAD_DIR)
    p.add_argument("--out", required=True)
    p.add_argument("--max-traces", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    df = load_metadata(a.stead_dir)
    if len(df) > a.max_traces:
        df = df.sample(a.max_traces, random_state=a.seed)
    writer, drops = StoreWriter(a.out), Counter()
    for h5_path, group in df.groupby("_h5"):
        with h5py.File(h5_path, "r") as h5:
            for i, r in enumerate(group.itertuples(), 1):
                got = read_trace(h5, r.trace_name)
                if got is None:
                    drops["dead_component"] += 1
                    continue
                wave, missing = got
                writer.add(f"stead/{r.trace_name}", wave, missing, {
                    "kind": "event", "split": station_split(r.network_code, r.receiver_code),
                    "source": "stead", "network": r.network_code,
                    "station": r.receiver_code, "event_id": r.source_id,
                    "magnitude": r.source_magnitude, "depth_km": r.source_depth_km,
                    "distance_km": r.source_distance_km, "start_time": r.trace_start_time,
                    "p_sample": float(r.p_arrival_sample), "p_source": "manual",
                    "p_tolerance_s": MANUAL_TOLERANCE_S,
                    "s_sample": float(r.s_arrival_sample) if r.s_status == "manual" else None,
                    "station_lat": r.receiver_latitude, "station_lon": r.receiver_longitude,
                    "event_lat": r.source_latitude, "event_lon": r.source_longitude,
                    "components": "ZNE", "back_azimuth_deg": r.back_azimuth_deg})
                if i % 10_000 == 0:
                    print(f"  {h5_path}: {i}/{len(group)}", flush=True)
    writer.close({"args": vars(a), "dropped": dict(drops)})
    print(f"wrote {len(writer.rows)} STEAD traces to {a.out}; dropped {dict(drops)}")


# ---------------------------------------------------------------------------
# validate-aic
# ---------------------------------------------------------------------------

def validate_aic(argv=None):
    p = argparse.ArgumentParser(prog="onset validate-aic")
    p.add_argument("--stead-dir", default=STEAD_DIR)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--offset-mean", type=float, default=-0.9,
                   help="Prediction minus truth, seconds. Negative: the prediction "
                        "is early, as TauP is on the KO pulls.")
    p.add_argument("--offset-sd", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    df = load_metadata(a.stead_dir).sample(a.n, random_state=a.seed)
    fs = SAMPLE_RATE
    rows = []
    for h5_path, group in df.groupby("_h5"):
        with h5py.File(h5_path, "r") as h5:
            for r in group.itertuples():
                got = read_trace(h5, r.trace_name)
                if got is None:
                    continue
                wave, missing = got
                truth = float(r.p_arrival_sample)
                guess = truth + rng.normal(a.offset_mean, a.offset_sd) * fs
                s = float(r.s_arrival_sample) if np.isfinite(r.s_arrival_sample) else None
                pick, ok, snr = refine_p(wave[:, 0], missing, guess, fs, s)
                rows.append({"ok": ok, "snr_db": r.snr_db, "magnitude": r.source_magnitude,
                             "guess_err_s": (guess - truth) / fs,
                             "pick_err_s": (pick - truth) / fs})
    res = pd.DataFrame(rows)
    acc = res[res.ok]
    q = [0.5, 0.68, 0.9, 0.95]

    def spread(e):
        return {f"|err| p{int(x * 100)}": round(float(np.abs(e).quantile(x)), 3) for x in q}

    print(f"traces {len(res)}  accepted {len(acc)} ({len(acc) / max(1, len(res)):.1%})")
    print(f"  prediction alone     median {res.guess_err_s.median():+.3f} s  {spread(res.guess_err_s)}")
    print(f"  AIC, accepted picks  median {acc.pick_err_s.median():+.3f} s  {spread(acc.pick_err_s)}")
    for t in (0.1, 0.2, 0.3, 0.5, 1.0):
        print(f"    within {t:.1f} s: {(acc.pick_err_s.abs() <= t).mean():.1%}")
    return res


if __name__ == "__main__":
    build()
