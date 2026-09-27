"""`onset train`: train the detector on one or more stores.

    onset train --data datasets/fdsn --out runs/fdsn_v1
    onset train --data datasets/fdsn --fallback datasets/stead --fallback-weight 0.3 ...

`--data` is the primary source: its events and noise are trained on, and its
validation split picks the checkpoint. `--fallback` stores add events only
(STEAD has no noise here), drawn for `--fallback-weight` of the event share of
each batch.

**Loss.** Weighted BCE on every token, plus `dt_weight` times smooth-L1 on
seconds-since-P for the tokens after P. Weights come from `labels.token_targets`.

**Model selection** is on latency, not loss: after each epoch the whole
validation split is scored, and the checkpoint with the best recall within 1 s
of P, at the threshold that keeps validation noise within
`fa_target_per_hour`, is kept as `best.pt`.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from onset import metrics
from onset.config import DataConfig, ModelConfig, TrainConfig, save_run_config
from onset.data import OnsetDataset
from onset.evaluate import predict, print_geometry, print_summary
from onset.model import OnsetDetector, count_parameters


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    for ds in getattr(info.dataset, "datasets", [info.dataset]):
        ds.reseed(info.seed % 2**32)


def check_noise(root, tcfg, fs: float = 100.0):
    """Stops before training on a store that cannot train or validate a
    detector, and warns about one that validates poorly.

    Without training noise the model only ever sees events and learns to fire
    on everything; without validation noise the false-trigger rate, and so
    the operating point and model selection, is undefined. With little
    validation noise one false trigger moves the operating point: the first
    wide build had 1.3 h, where one trigger is 0.76 per hour. And an event
    trace whose station has no noise gets neither a lead-in splice nor a
    station context, which brings back the "data just began" cue.
    """
    idx = pd.read_csv(Path(root) / "index.csv", low_memory=False)
    noise = idx[idx.kind == "noise"]
    hours = {s: noise.loc[noise.split == s, "n_samples"].sum() / fs / 3600
             for s in ("train", "val", "test")}
    counts = {s: int((noise.split == s).sum()) for s in ("train", "val", "test")}
    print(f"  noise: " + ", ".join(f"{s} {counts[s]} traces ({hours[s]:.1f} h)"
                                    for s in counts))
    problems = []
    if tcfg.noise_fraction > 0 and not counts["train"]:
        problems.append("no training noise: the model would learn that everything is an event")
    if not counts["val"]:
        problems.append("no validation noise: false triggers per hour, the operating point "
                        "and model selection are undefined")
    if problems:
        report = Path(root) / "build.json"
        dropped = {}
        if report.exists():
            dropped = {k: v for k, v in json.loads(report.read_text()).get("dropped", {}).items()
                       if k.startswith(("noise", "context"))}
        raise SystemExit(
            f"{root}: " + "; ".join(problems) + ".\n"
            f"  build.json drops for noise and context: {dropped or 'none recorded'}\n"
            "  See docs/MANUAL.md section 2: --context-dir and --noise-dir must point at the "
            "noise pulled with the events, and --noise-seconds must fit the pulled window.")
    if hours["val"] < 5.0:
        print(f"  warning: {hours['val']:.1f} h of validation noise; one false trigger is "
              f"{1 / hours['val']:.2f}/h against a budget of {tcfg.fa_target_per_hour}/h, "
              "so the operating point and model selection will be noisy")
    ev = idx[(idx.kind == "event") & (idx.split == "train")]
    # Station context is what tells the model what this station normally looks
    # like, and sets the input scale; without it every trace uses the learned
    # null context and the scale of its own first second.
    contexts = set(idx.loc[idx.kind == "context", "key"])
    with_ctx = ev.context_key.isin(contexts).mean() if len(ev) else 1.0
    print(f"  context: {len(contexts)} traces; {with_ctx:.0%} of training event traces have one")
    if with_ctx < 0.5:
        report = Path(root) / "build.json"
        short = (json.loads(report.read_text()).get("dropped", {}).get("context_short", 0)
                 if report.exists() else 0)
        print(f"  warning: most event traces have no station context"
              + (f"; build.json dropped {short} as context_short: --context-seconds is "
                 "longer than the pulled window (290 for 5 min pulls)" if short else ""))
    have = set(map(tuple, idx.loc[idx.kind.isin(("noise", "context")),
                                  ["network", "station"]].drop_duplicates().to_numpy()))
    covered = (np.mean([(n, s) in have for n, s in zip(ev.network, ev.station)])
               if len(ev) else 1.0)
    if covered < 0.8:
        print(f"  warning: only {covered:.0%} of training event traces have noise from their own "
              "station, for the lead-in splice and station context; the rest start at origin")


def training_set(primary, fallback, dcfg, mcfg, tcfg):
    """ConcatDataset plus per-example sampling weights that give each group
    its share of a batch regardless of how large the group is."""
    groups = [(OnsetDataset(primary, "train", dcfg, mcfg, True, ("noise",)),
               tcfg.noise_fraction)]
    ev_share = 1 - tcfg.noise_fraction
    fb_share = tcfg.fallback_weight if fallback else 0.0
    groups.append((OnsetDataset(primary, "train", dcfg, mcfg, True, ("event",)),
                   ev_share * (1 - fb_share)))
    for fb in fallback:
        groups.append((OnsetDataset(fb, "train", dcfg, mcfg, True, ("event",)),
                       ev_share * fb_share / len(fallback)))
    groups = [(ds, share) for ds, share in groups if len(ds) and share > 0]
    total = sum(s for _, s in groups)
    weights = np.concatenate([np.full(len(ds), share / total / len(ds)) for ds, share in groups])
    for ds, share in groups:
        print(f"  train {ds.root.name:>12s} {ds.rows.kind.iloc[0]:>5s}: "
              f"{len(ds):7d} traces, {share / total:.0%} of draws")
    return ConcatDataset([ds for ds, _ in groups]), torch.as_tensor(weights)


def lr_at(step, tcfg):
    total = tcfg.epochs * tcfg.steps_per_epoch
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    frac = (step - tcfg.warmup_steps) / max(1, total - tcfg.warmup_steps)
    return tcfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, frac)))


def geometry_loss(out, b):
    """Gaussian NLL on log distance and von Mises NLL on back-azimuth, over the
    tokens where dt is trained (after P, outside the label's tolerance) and,
    with a second event added, before its P: the targets are the first event's."""
    m = b.get("geo_mask", b["dt_mask"])
    d_ok = torch.isfinite(b["dist_km"])[:, None] & (m > 0)
    y = torch.log(torch.nan_to_num(b["dist_km"], nan=1.0).clamp_min(1.0))[:, None]
    mu, lv = out["log_dist"].float(), out["log_dist_var"].float()
    nll_d = 0.5 * (lv + (y - mu) ** 2 * torch.exp(-lv))
    loss_d = (nll_d * d_ok).sum() / d_ok.sum().clamp_min(1)

    b_ok = torch.isfinite(b["baz_rad"])[:, None] & (m > 0)
    th = torch.nan_to_num(b["baz_rad"], nan=0.0)[:, None]
    v = out["baz_vec"].float()
    cos_err = (v[..., 0] * torch.sin(th) + v[..., 1] * torch.cos(th)) / (v.norm(dim=-1) + 1e-6)
    kappa = torch.exp(out["baz_log_kappa"].float())
    nll_b = -kappa * cos_err + torch.log(torch.special.i0e(kappa)) + kappa
    loss_b = (nll_b * b_ok).sum() / b_ok.sum().clamp_min(1)
    return loss_d, loss_b


def loss_fn(out, b, dt_weight, geo_weight=0.0):
    w = b["w"]
    bce = F.binary_cross_entropy_with_logits(out["logit"].float(), b["y"], weight=w,
                                             reduction="sum") / w.sum().clamp_min(1)
    m = b["dt_mask"] * b["dt_w"] if "dt_w" in b else b["dt_mask"]
    dt = (F.smooth_l1_loss(out["dt"].float(), b["dt"], reduction="none") * m).sum() \
        / m.sum().clamp_min(1)
    loss = bce + dt_weight * dt
    if "log_dist" in out and geo_weight > 0:
        ld, lb = geometry_loss(out, b)
        loss = loss + geo_weight * (ld + lb)
    return loss, bce, dt


def add_config_flags(p, cls, prefix=""):
    for f in fields(cls):
        if f.type in ("int", "float", "str") or f.type in (int, float, str):
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{prefix}{f.name.replace('_', '-')}", dest=f"{prefix}{f.name}",
                           type=typ, default=None)


def apply_flags(cfg, args, prefix=""):
    for f in fields(cfg):
        v = getattr(args, f"{prefix}{f.name}", None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(prog="onset train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Primary store (build-fdsn output).")
    p.add_argument("--fallback", action="append", default=[],
                   help="Extra event-only store, e.g. build-stead output. Repeatable.")
    p.add_argument("--out", required=True)
    g = p.add_argument_group("model (ModelConfig)")
    add_config_flags(g, ModelConfig)
    g = p.add_argument_group("data (DataConfig)")
    add_config_flags(g, DataConfig)
    g = p.add_argument_group("training (TrainConfig)")
    add_config_flags(g, TrainConfig)
    a = p.parse_args(argv)

    mcfg = apply_flags(ModelConfig(), a)
    dcfg = apply_flags(DataConfig(), a)
    tcfg = apply_flags(TrainConfig(), a)
    seed_everything(tcfg.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_run_config(out, mcfg, dcfg, tcfg, sources={"data": a.data, "fallback": a.fallback})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    model = OnsetDetector(mcfg).to(device)
    print(f"model: {count_parameters(model):,} parameters | token {mcfg.token_seconds:.2f} s | "
          f"window {mcfg.window_tokens * mcfg.token_seconds:.0f} s/layer | "
          f"lookback {mcfg.lookback_tokens * mcfg.token_seconds:.1f} s")

    check_noise(a.data, tcfg, mcfg.sample_rate)
    train_ds, weights = training_set(a.data, a.fallback, dcfg, mcfg, tcfg)
    sampler = WeightedRandomSampler(weights, tcfg.steps_per_epoch * tcfg.batch_size,
                                    replacement=True)
    loader = DataLoader(train_ds, batch_size=tcfg.batch_size, sampler=sampler,
                        num_workers=tcfg.num_workers, worker_init_fn=worker_init,
                        pin_memory=amp, persistent_workers=tcfg.num_workers > 0,
                        drop_last=True)
    val_ds = OnsetDataset(a.data, "val", dcfg, mcfg, train=False)
    rule = metrics.TriggerRule.from_config(tcfg, mcfg)
    print(f"  val {len(val_ds)} traces")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best, step = -1.0, 0
    history = open(out / "history.jsonl", "a")
    for epoch in range(tcfg.epochs):
        model.train()
        t0, sums, n = time.time(), np.zeros(3), 0
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            for group in opt.param_groups:
                group["lr"] = lr_at(step, tcfg)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                o = model(b["x"], b["ctx"], b["has_ctx"])
            loss, bce, dtl = loss_fn(o, b, tcfg.dt_weight, tcfg.geo_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: bce {bce.item()} "
                                   f"dt {dtl.item()}, max|x| {b['x'].abs().max().item():.3g}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums += [loss.item(), bce.item(), dtl.item()]
            n += 1
            step += 1

        events, noise = predict(model, val_ds, device, workers=tcfg.num_workers)
        rows = metrics.sweep(events, noise, mcfg.stride, mcfg.sample_rate, rule=rule)
        s = metrics.summary(rows, tcfg.fa_target_per_hour, metrics.exact_operating_point(
            rows, events, noise, mcfg.stride, mcfg.sample_rate, tcfg.fa_target_per_hour, rule))
        loss_avg = sums / max(1, n)
        geo = metrics.geometry_table(events, mcfg.stride, mcfg.sample_rate) if mcfg.geometry else []
        rec = {"epoch": epoch + 1, "step": step, "loss": loss_avg[0], "bce": loss_avg[1],
               "dt_loss": loss_avg[2], "seconds": time.time() - t0, "val": s, "geometry": geo}
        history.write(json.dumps(rec, default=float) + "\n")
        history.flush()
        print(f"epoch {epoch + 1:3d}  loss {loss_avg[0]:.4f} (bce {loss_avg[1]:.4f} "
              f"dt {loss_avg[2]:.3f})  val recall@1s {s['recall@1.0s']:.3f} "
              f"@thr {s['threshold']:.5f} ({s['false_per_hour']:.2f} FA/h)  "
              f"lat p50 {s['latency_p50_s']:.2f}s"
              + (f"  second@2s {s['second_recall@2.0s']:.3f} "
                 f"(dt min {s.get('second_dt_min_p50', float('nan')):.1f}s)  score {s['score']:.3f}"
                 if s.get("second_n") else "")
              + f"  {time.time() - t0:.0f}s"
              + "".join(f"  | {g['window'].split(' after')[0]}: {g['dist_abs_err_km_p50']:.1f} km"
                        f" {g['baz_err_deg_p50']:.0f}°" for g in geo
                        if g["window"] in ("1-2 s after P", "after S")), flush=True)
        torch.save(model.state_dict(), out / "last.pt")
        if s["score"] > best:
            best = s["score"]
            torch.save(model.state_dict(), out / "best.pt")
            (out / "val_best.json").write_text(json.dumps(
                {"epoch": epoch + 1, "summary": s, "sweep": rows, "geometry": geo},
                indent=2, default=float))
            print(f"          -> best.pt (score {best:.3f}: recall within 1 s over "
                  f"first and second onsets)")

    hours = sum((~x["missing_tokens"]).sum() for x in noise) * mcfg.token_seconds / 3600
    best_rec = json.loads((out / "val_best.json").read_text())
    print_summary(f"best epoch {best_rec['epoch']} on val", best_rec["summary"],
                  len(events), hours)
    print_geometry(best_rec.get("geometry", []))


if __name__ == "__main__":
    main()
