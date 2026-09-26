"""The on-disk dataset: one HDF5 of waveforms plus `index.csv`.

    <root>/waveforms.h5    wave/<key>     (T, 3) float32, Z N E, causally filtered counts
                           missing/<key>  (T,) uint8, only for traces with a gap
    <root>/index.csv       one row per trace (columns below)
    <root>/build.json      how it was built, and what was dropped and why

Every source (FDSN pulls, STEAD) is written in this one format, so training
reads all of them the same way.
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WAVEFORMS = "waveforms.h5"
INDEX = "index.csv"

COLUMNS = [
    "key", "kind", "split", "source", "network", "station", "event_id",
    "magnitude", "depth_km", "distance_km", "start_time", "n_samples",
    "missing_fraction", "p_sample", "p_source", "p_tolerance_s",
    "p_predicted_sample", "s_sample", "pick_snr", "context_key",
]
KINDS = ("event", "noise", "context")


def station_split(network: str, station: str, val: int = 10, test: int = 10) -> str:
    """Deterministic station-disjoint split: the same station lands in the same
    split in every source and every rebuild."""
    h = zlib.crc32(f"{network}.{station}".encode()) % 100
    return "test" if h < test else "val" if h < test + val else "train"


class StoreWriter:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.root / WAVEFORMS, "w")
        self.rows: list[dict] = []

    def add(self, key: str, wave: np.ndarray, missing: np.ndarray, meta: dict):
        self.h5.create_dataset(f"wave/{key}", data=wave.astype(np.float32))
        if missing.any():
            self.h5.create_dataset(f"missing/{key}", data=missing.astype(np.uint8))
        self.rows.append({**{c: None for c in COLUMNS}, **meta, "key": key,
                          "n_samples": len(wave),
                          "missing_fraction": float(missing.mean())})

    def close(self, report: dict | None = None):
        self.h5.close()
        pd.DataFrame(self.rows, columns=COLUMNS).to_csv(self.root / INDEX, index=False)
        if report is not None:
            (self.root / "build.json").write_text(json.dumps(report, indent=2, default=str))


class StoreReader:
    """Read access that survives DataLoader workers: h5py handles are not
    fork-safe, so each process opens its own on first use."""

    def __init__(self, root):
        self.root = Path(root)
        self.index = pd.read_csv(self.root / INDEX, low_memory=False)
        self._h5 = None
        self._pid = None

    @property
    def h5(self):
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.root / WAVEFORMS, "r")
            self._pid = os.getpid()
        return self._h5

    def read(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        wave = self.h5[f"wave/{key}"][()]
        m = self.h5.get(f"missing/{key}")
        missing = m[()].astype(bool) if m is not None else np.zeros(len(wave), bool)
        return wave, missing
