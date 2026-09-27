"""`onset evaluate`: latency and false triggers of a trained run on one split.

    onset evaluate runs/fdsn_v1 --data datasets/fdsn --split test

Prints the operating point (the lowest threshold within the false-trigger
budget) and writes, into the run directory:

    eval_<dataset>_<split>.json    the threshold sweep and the operating point
    eval_<dataset>_<split>.csv     one row per event trace at the operating point
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from onset import metrics
from onset.config import DataConfig, ModelConfig, load_run_config
from onset.data import OnsetDataset, pad_collate
from onset.model import OnsetDetector


def length_batches(ds: OnsetDataset, batch_size: int) -> list[list[int]]:
    """Whole traces differ in length by kind; batch equal stored lengths
    together, so a batch pads by at most the splice."""
    by_len = defaultdict(list)
    for i, n in enumerate(ds.rows.n_samples.to_numpy()):
        by_len[int(n)].append(i)
    return [ix[k:k + batch_size] for ix in by_len.values()
            for k in range(0, len(ix), batch_size)]


@torch.no_grad()
def predict(model: OnsetDetector, ds: OnsetDataset, device, batch_size=64, workers=4):
    """Runs whole traces. Returns (events, noise) lists for `metrics.sweep`."""
    model.eval()
    loader = DataLoader(ds, batch_sampler=length_batches(ds, batch_size),
                        num_workers=workers, collate_fn=pad_collate)
    stride = model.cfg.stride
    events, noise = [], []
    amp = device.type == "cuda"
    for b in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["x"].to(device), b["ctx"].to(device), b["has_ctx"].to(device))
        p = torch.sigmoid(out["logit"].float()).cpu().numpy()
        dt = out["dt"].float().cpu().numpy()
        geo = None
        if "log_dist" in out:
            v = out["baz_vec"].float()
            geo = {"log_dist": out["log_dist"].float().cpu().numpy(),
                   "log_dist_var": out["log_dist_var"].float().cpu().numpy(),
                   "baz": torch.atan2(v[..., 0], v[..., 1]).cpu().numpy(),
                   "baz_log_kappa": out["baz_log_kappa"].float().cpu().numpy()}
        x = b["x"].numpy()
        n_tok = p.shape[1]
        miss = x[:, : n_tok * stride, 3].reshape(len(x), n_tok, stride).max(-1) > 0
        for k in range(len(p)):
            n = int(b["n_tokens"][k])
            item = {"p": p[k, :n], "dt": dt[k, :n], "missing_tokens": miss[k, :n],
                    "index": int(b["index"][k]), "has_ctx": bool(b["has_ctx"][k])}
            if geo is not None:
                item.update({g: v[k, :n] for g, v in geo.items()})
            if bool(b["is_event"][k]):
                events.append({**item, "p_s": float(b["p_s"][k]), "tol_s": float(b["tol_s"][k]),
                               "dist_km": float(b["dist_km"][k]), "baz_rad": float(b["baz_rad"][k]),
                               "s_s": float(b["s_s"][k]), "p2_s": float(b["p2_s"][k]),
                               "tol2_s": float(b["tol2_s"][k])})
            else:
                noise.append(item)
    return events, noise


def per_trace(events, ds, stride, fs, thr, release_ratio=0.5, rule=None) -> pd.DataFrame:
    rows = []
    for e in events:
        lat, early, onset = metrics.score_event(e["p"], e["dt"], e["p_s"], e["tol_s"],
                                                stride, fs, thr, thr * release_ratio, rule)
        r = ds.rows.iloc[e["index"]]
        rows.append({"key": r.key, "station": r.station, "magnitude": r.magnitude,
                     "distance_km": r.distance_km, "p_source": r.p_source,
                     "has_ctx": e["has_ctx"], "latency_s": lat, "early": early,
                     "onset_err_s": onset, "p_max": float(e["p"].max())})
    return pd.DataFrame(rows)


def load_model(run_dir, device, which="best.pt"):
    mcfg, dcfg, tcfg, extra = load_run_config(run_dir)
    model = OnsetDetector(mcfg).to(device)
    model.load_state_dict(torch.load(Path(run_dir) / which, map_location=device,
                                     weights_only=True))
    return model.eval(), mcfg, dcfg, tcfg


def print_summary(title: str, s: dict, n_events: int, noise_hours: float):
    print(f"\n{title}")
    print(f"  events {n_events}   noise {noise_hours:.1f} h   "
          f"false-trigger budget {s['fa_target_per_hour']}/h")
    print(f"  threshold {s['threshold']:.6f}   false triggers {s['false_per_hour']:.2f}/h   "
          f"early (pre-P) triggers on events {s['early_rate']:.1%}")
    print("  recall  " + "  ".join(f"≤{d:g}s {s[f'recall@{d}s']:.1%}" for d in metrics.DELAYS_S)
          + f"   ever {s['detected']:.1%}")
    if s.get("second_n"):
        print(f"  second onsets in the coda ({s['second_n']}): recall  "
              + "  ".join(f"≤{d:g}s {s[f'second_recall@{d}s']:.1%}" for d in metrics.DELAYS_S)
              + f"   both onsets ≤1s {s['onset_recall@1.0s']:.1%}")
    print(f"  median latency {s['latency_p50_s']:.2f} s   "
          f"median onset error from dt {s['onset_abs_err_p50_s']:.2f} s")


def print_geometry(rows):
    if not rows:
        return
    print("\n  where is it (geometry head), per event, median over its tokens")
    print(f"    {'window':<22s} {'n':>5} {'dist err km':>11} {'dist err %':>10} "
          f"{'within 1 sd':>11} {'baz err deg':>11}")
    for r in rows:
        print(f"    {r['window']:<22s} {r['n']:>5} {r['dist_abs_err_km_p50']:>11.1f} "
              f"{100 * r['dist_rel_err_p50']:>9.0f}% {r['cal_1sd']:>11.0%} "
              f"{r['baz_err_deg_p50']:>11.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="onset evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--fa-target", type=float, default=None,
                   help="False triggers per hour; defaults to the run's own.")
    p.add_argument("--no-context", action="store_true",
                   help="Score as a station with no baseline yet.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg, dcfg, tcfg = load_model(a.run_dir, device, a.checkpoint)
    ds = OnsetDataset(a.data, a.split, dcfg, mcfg, train=False)
    if a.no_context:
        ds.contexts = set()        # every trace then falls back to the null context
    events, noise = predict(model, ds, device, a.batch_size, a.workers)
    fa = a.fa_target if a.fa_target is not None else tcfg.fa_target_per_hour
    rule = metrics.TriggerRule.from_config(tcfg, mcfg)
    rows = metrics.sweep(events, noise, mcfg.stride, mcfg.sample_rate, rule=rule)
    s = metrics.summary(rows, fa, metrics.exact_operating_point(
        rows, events, noise, mcfg.stride, mcfg.sample_rate, fa, rule))
    hours = sum((~n["missing_tokens"]).sum() for n in noise) * mcfg.token_seconds / 3600

    name = f"{Path(a.data).name}_{a.split}" + ("_noctx" if a.no_context else "")
    print_summary(f"{a.run_dir} on {name}", s, len(events), hours)
    geo_rows = metrics.geometry_table(events, mcfg.stride, mcfg.sample_rate)
    print_geometry(geo_rows)
    table = per_trace(events, ds, mcfg.stride, mcfg.sample_rate, s["threshold"], rule=rule)
    if len(table):
        table["mag_bin"] = pd.cut(table.magnitude, [-9, 2, 3, 4, 5, 10],
                                  labels=["<2", "2-3", "3-4", "4-5", "≥5"])
        by = table.groupby(["mag_bin"], observed=True).agg(
            n=("key", "size"), recall_1s=("latency_s", lambda v: (v <= 1).mean()),
            latency_p50=("latency_s", "median"))
        print("\n  by magnitude (at the operating threshold)\n" + by.round(3).to_string())
        by_src = table.groupby("p_source").agg(
            n=("key", "size"), recall_1s=("latency_s", lambda v: (v <= 1).mean()),
            latency_p50=("latency_s", "median"))
        print("\n  by label source\n" + by_src.round(3).to_string())
    out = Path(a.run_dir)
    (out / f"eval_{name}.json").write_text(json.dumps(
        {"summary": s, "sweep": rows, "n_events": len(events), "noise_hours": hours,
         "geometry": geo_rows},
        indent=2, default=float))
    table.to_csv(out / f"eval_{name}.csv", index=False)
    print(f"\n  -> {out / f'eval_{name}.json'}")


if __name__ == "__main__":
    main()
