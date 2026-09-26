"""`onset replay`: run a trained detector over continuous miniSEED, as deployed.

    onset replay runs/fdsn_v1 day_before_24h/event_543430_raw.mseed \\
        window_post_60s/event_543430_raw.mseed --out replays/kahramanmaras

All files are read as one recording, so a day of lead-in and the event window
that follows it join into one stream. Each station is then:

1. filtered causally, with a restart after every gap (`dsp`);
2. conditioned with a **station state that is updated online**, as a
   deployment would do it. The stream starts with no context and a scale taken
   from its first second. After every block (`--block-seconds`) whose outputs
   all stayed below `--quiet` and whose data is complete, the last
   `ctx_seconds` become the new context and its RMS the new scale;
3. scored in overlapping blocks, which gives the streaming outputs up to one
   approximation: a context refresh applies to a block's lead-in as well as
   its new tokens. Scale changes are exact, since each sample is conditioned
   with the scale that was in force when it arrived.

With `--catalog` and `--stations`, visible catalogued arrivals are predicted
with TauP, refined with AIC, and matched to triggers:

- **detected**: the first trigger in `[P - tol, P + 10 s]`, with its latency;
- **unmatched**: a trigger with no arrival in `[-tol, +60 s]` around it (the
  60 s absorbs S and coda re-triggers). Unmatched is not necessarily false:
  the catalogue misses small events.

Writes `<out>/<NET.STA>.npz` (per-token time, p, dt), `<NET.STA>_triggers.csv`,
and `summary.json`. A model with the geometry head adds per-token `dist_km`,
`log_dist_sd`, `baz_deg` and `kappa` to the npz, and those values at each
trigger and 10 s after it to the triggers CSV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import Stream, read

from onset import metrics
from onset.build_fdsn import instruments, to_grid
from onset.catalog import S_PHASES, Catalog, TravelTimes, distance_km, load_stations
from onset.conditioning import channel_scale, condition
from onset.dsp import filter_components
from onset.evaluate import load_model
from onset.labels import refine_p
from onset.stream import block_overlap_tokens


GEO_AFTER_S = 10.0                 # second geometry reading after each trigger


class ScaleSchedule:
    """The per-component scale in force at each sample: piecewise constant,
    changing only at block ends."""

    def __init__(self, first_scale):
        self.starts, self.scales = [0], [first_scale]

    def set(self, at_sample, scale):
        self.starts.append(at_sample)
        self.scales.append(scale)

    def condition(self, wave, missing, a, b):
        out = np.empty((b - a, 4), np.float32)
        edges = self.starts[1:] + [b]
        for start, end, scale in zip(self.starts, edges, self.scales):
            lo, hi = max(a, start), min(b, end)
            if lo < hi:
                out[lo - a: hi - a] = condition(wave[lo:hi], missing[lo:hi], scale)
        return out


@torch.no_grad()
def replay_station(model, dcfg, wave, missing, block_s=60.0, quiet=0.3):
    """Scores one station's continuous (T, 3) filtered recording.

    Returns:
        (probabilities, dt, context refresh sample indices, geometry), the
        last a dict of per-token `dist_km`, `log_dist_sd`, `baz_deg` and
        `kappa` for a model with the geometry head, else None.
    """
    cfg = model.cfg
    s, fs = cfg.stride, cfg.sample_rate
    dev = next(model.parameters()).device
    n_tok = len(wave) // s
    block = int(block_s * fs) // s
    lead = block_overlap_tokens(model)
    ctx_len = int(round(dcfg.ctx_seconds * fs / s)) * s
    n0 = int(dcfg.fallback_scale_s * fs)
    sched = ScaleSchedule(channel_scale(wave[:n0], missing[:n0]))
    c = has = None
    refreshes = []
    probs = np.zeros(n_tok, np.float32)
    dts = np.zeros(n_tok, np.float32)
    geo = ({k: np.full(n_tok, np.nan, np.float32)
            for k in ("dist_km", "log_dist_sd", "baz_deg", "kappa")} if cfg.geometry else None)
    for t0 in range(0, n_tok, block):
        t1 = min(t0 + block, n_tok)
        a = max(0, t0 - lead)
        x = torch.as_tensor(sched.condition(wave, missing, a * s, t1 * s), device=dev)[None]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(x, c, has)
        probs[t0:t1] = torch.sigmoid(out["logit"][0, t0 - a:].float()).cpu().numpy()
        dts[t0:t1] = out["dt"][0, t0 - a:].float().cpu().numpy()
        if geo is not None:
            g = {k: v[0, t0 - a:].float().cpu() for k, v in out.items() if k != "logit"}
            geo["dist_km"][t0:t1] = g["log_dist"].exp().numpy()
            geo["log_dist_sd"][t0:t1] = (0.5 * g["log_dist_var"]).exp().numpy()
            geo["baz_deg"][t0:t1] = np.degrees(torch.atan2(g["baz_vec"][:, 0],
                                                           g["baz_vec"][:, 1]).numpy()) % 360
            geo["kappa"][t0:t1] = g["baz_log_kappa"].exp().numpy()

        end = t1 * s
        ctx_tok = ctx_len // s
        if end >= ctx_len and t1 >= ctx_tok and probs[t1 - ctx_tok: t1].max() < quiet \
                and missing[end - ctx_len: end].mean() < 0.05:
            scale = channel_scale(wave[end - ctx_len: end], missing[end - ctx_len: end])
            sched.set(end, scale)
            ctx = condition(wave[end - ctx_len: end], missing[end - ctx_len: end], scale)
            c = torch.as_tensor(ctx, device=dev)[None]
            has = torch.tensor([True], device=dev)
            refreshes.append(end)
    return probs, dts, refreshes, geo


def match(arrivals, trig_t, tol, detect_window=10.0, coda=60.0):
    """(per-arrival latency or nan, unmatched trigger count)."""
    lat = []
    for p in arrivals:
        hit = trig_t[(trig_t >= p - tol) & (trig_t <= p + detect_window)]
        lat.append(hit[0] - p if len(hit) else np.nan)
    unmatched = sum(1 for t in trig_t
                    if not any(p - tol <= t <= p + coda for p in arrivals))
    return np.asarray(lat), unmatched


def main(argv=None):
    p = argparse.ArgumentParser(prog="onset replay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--threshold", type=float, default=None,
                   help="Defaults to the run's validation operating point.")
    p.add_argument("--block-seconds", type=float, default=60.0)
    p.add_argument("--quiet", type=float, default=0.3)
    p.add_argument("--catalog", default=None)
    p.add_argument("--stations", default=None)
    p.add_argument("--tolerance", type=float, default=0.5,
                   help="Seconds a trigger may precede the refined P and still count.")
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg, dcfg, tcfg = load_model(a.run_dir, device, a.checkpoint)
    thr = a.threshold
    if thr is None:
        thr = json.loads((Path(a.run_dir) / "val_best.json").read_text())["summary"]["threshold"]
    release = thr * 0.5
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cat = Catalog(a.catalog) if a.catalog else None
    stations = load_stations(a.stations) if a.stations else {}
    taup = TravelTimes()

    st = Stream()
    for f in a.files:
        st += read(f)
    summary = {"threshold": thr, "files": a.files, "stations": {}}
    fs, s = mcfg.sample_rate, mcfg.stride
    for (net, sta), comps in sorted(instruments(st).items()):
        t_start = min(tr.stats.starttime for c in comps for tr in c)
        t_end = max(tr.stats.endtime for c in comps for tr in c)
        n = int(round((t_end - t_start) * fs)) + 1
        n -= n % s
        raw, miss3 = np.zeros((n, 3)), np.ones((n, 3), bool)
        for i, c in enumerate(comps):
            raw[:, i], miss3[:, i] = to_grid(c, t_start, n)
        wave, missing = filter_components(raw, miss3)
        probs, dts, refreshes, geo = replay_station(model, dcfg, wave, missing,
                                               a.block_seconds, a.quiet)
        t = metrics.token_times(len(probs), s, fs)
        edges = metrics.rising_edges(probs, thr, release)
        name = f"{net}.{sta}"
        np.savez_compressed(out / f"{name}.npz", t=t, p=probs, dt=dts,
                            start=str(t_start), refreshes=np.asarray(refreshes),
                            **(geo or {}))
        trig = pd.DataFrame({"time_s": t[edges], "utc": [str(t_start + x) for x in t[edges]],
                             "p": probs[edges], "onset_s": t[edges] - dts[edges]})
        if geo is not None:
            # Where the event is as seen from here, at the trigger and GEO_AFTER_S
            # later (ayzek relocates as these sharpen).
            for tag, k in [("", edges),
                           (f"_{GEO_AFTER_S:g}s", np.minimum(edges + int(GEO_AFTER_S * fs / s),
                                                              len(probs) - 1))]:
                for g, v in geo.items():
                    trig[f"{g}{tag}"] = v[k]
        hours = (~missing).sum() / fs / 3600
        res = {"hours": hours, "missing_fraction": float(missing.mean()),
               "triggers": len(edges), "context_refreshes": len(refreshes)}

        coords = stations.get((net, sta)) or stations.get(sta)
        if cat is not None and coords is not None:
            t0 = float(t_start.timestamp)
            arr = cat.arrivals(coords[0], coords[1], t0, t0 + n / fs, taup)
            p_s = []
            for eid, t_arr in arr:
                ev = cat.event(eid)
                dist = distance_km(ev.lat, ev.lon, *coords)
                ts = taup.first(dist, ev.depth_km, S_PHASES)
                guess = (t_arr - t0) * fs
                s_guess = (ev.origin + ts - t0) * fs if ts is not None else None
                pick, ok, _ = refine_p(wave[:, 0], missing, guess, fs, s_guess)
                p_s.append(pick / fs)
            lat, unmatched = match(np.asarray(p_s), t[edges], a.tolerance)
            trig["matched"] = [any(ps - a.tolerance <= x <= ps + 60 for ps in p_s)
                               for x in t[edges]]
            res.update({"catalogued_arrivals": len(arr),
                        "detected": int(np.isfinite(lat).sum()),
                        "latency_p50_s": float(np.nanmedian(lat)) if np.isfinite(lat).any() else None,
                        **{f"recall@{d}s": float(np.mean(lat <= d)) if len(lat) else None
                           for d in metrics.DELAYS_S},
                        "unmatched_triggers": unmatched,
                        "unmatched_per_hour": unmatched / hours if hours else None})
        trig.to_csv(out / f"{name}_triggers.csv", index=False)
        summary["stations"][name] = res
        print(f"{name}: {hours:.2f} h, {res['triggers']} triggers, "
              f"{res['context_refreshes']} context refreshes"
              + (f", {res['detected']}/{res['catalogued_arrivals']} catalogued arrivals "
                 f"detected, median latency {res['latency_p50_s']}, "
                 f"{res['unmatched_per_hour']:.2f} unmatched/h"
                 if "detected" in res else ""))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
