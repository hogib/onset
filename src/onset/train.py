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
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from onset import metrics
from onset.config import DataConfig, ModelConfig, TrainConfig, save_run_config
from onset.data import OnsetDataset
from onset.evaluate import predict, print_summary
from onset.model import OnsetDetector, count_parameters


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    for ds in getattr(info.dataset, "datasets", [info.dataset]):
        ds.reseed(info.seed % 2**32)


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


def loss_fn(out, b, dt_weight):
    w = b["w"]
    bce = F.binary_cross_entropy_with_logits(out["logit"].float(), b["y"], weight=w,
                                             reduction="sum") / w.sum().clamp_min(1)
    m = b["dt_mask"]
    dt = (F.smooth_l1_loss(out["dt"].float(), b["dt"], reduction="none") * m).sum() \
        / m.sum().clamp_min(1)
    return bce + dt_weight * dt, bce, dt


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

    train_ds, weights = training_set(a.data, a.fallback, dcfg, mcfg, tcfg)
    sampler = WeightedRandomSampler(weights, tcfg.steps_per_epoch * tcfg.batch_size,
                                    replacement=True)
    loader = DataLoader(train_ds, batch_size=tcfg.batch_size, sampler=sampler,
                        num_workers=tcfg.num_workers, worker_init_fn=worker_init,
                        pin_memory=amp, persistent_workers=tcfg.num_workers > 0,
                        drop_last=True)
    val_ds = OnsetDataset(a.data, "val", dcfg, mcfg, train=False)
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
            loss, bce, dtl = loss_fn(o, b, tcfg.dt_weight)
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
        rows = metrics.sweep(events, noise, mcfg.stride, mcfg.sample_rate)
        s = metrics.summary(rows, tcfg.fa_target_per_hour)
        loss_avg = sums / max(1, n)
        rec = {"epoch": epoch + 1, "step": step, "loss": loss_avg[0], "bce": loss_avg[1],
               "dt_loss": loss_avg[2], "seconds": time.time() - t0, "val": s}
        history.write(json.dumps(rec, default=float) + "\n")
        history.flush()
        print(f"epoch {epoch + 1:3d}  loss {loss_avg[0]:.4f} (bce {loss_avg[1]:.4f} "
              f"dt {loss_avg[2]:.3f})  val recall@1s {s['recall@1.0s']:.3f} "
              f"@thr {s['threshold']:.3f} ({s['false_per_hour']:.2f} FA/h)  "
              f"lat p50 {s['latency_p50_s']:.2f}s  {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), out / "last.pt")
        if s["score"] > best:
            best = s["score"]
            torch.save(model.state_dict(), out / "best.pt")
            (out / "val_best.json").write_text(json.dumps(
                {"epoch": epoch + 1, "summary": s, "sweep": rows}, indent=2, default=float))
            print(f"          -> best.pt (recall@1s {best:.3f})")

    hours = sum((~x["missing_tokens"]).sum() for x in noise) * mcfg.token_seconds / 3600
    best_rec = json.loads((out / "val_best.json").read_text())
    print_summary(f"best epoch {best_rec['epoch']} on val", best_rec["summary"],
                  len(events), hours)


if __name__ == "__main__":
    main()
