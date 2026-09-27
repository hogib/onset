"""`onset audit-noise`: are the false triggers on noise real earthquakes?

    onset audit-noise runs/fdsn_wide_ctx --data datasets/fdsn_wide --split val

The builder rejects a noise window only for *catalogued* events. An event
below the catalogue's completeness stays in the noise, labelled noise, and a
model that detects it is scored as a false trigger; a noise set full of them
forces the operating threshold up against 1 and makes recall at that
threshold jump from epoch to epoch.

The test uses what makes a real event different from a detector's own
blunder: it is seen at several stations. The noise windows pulled for one
catalogue event cover the same clock time at every station, so a trigger at
one station and another at a second station within the P travel time
between them (plus `--slack`) is a coincidence. The split is by station, so
the other stations of a pull are in other splits: the windows of the pulls
with a trigger are scored at every station, whatever its split, and only
the audited split's triggers are counted. Chance coincidences are
estimated by sliding each station's triggers around its own window, which
keeps every station's trigger count and destroys real moveout. Coincidences
well above chance mean the noise holds earthquakes.

Writes `<run>/audit_noise_<split>.csv`: every trigger, with whether it
coincides with one at another station and the pull it belongs to.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from onset import metrics
from onset.catalog import haversine_km
from onset.data import OnsetDataset

VP_KM_S = 6.0


def coincidences(trig: pd.DataFrame, slack_s: float = 2.0,
                 shifts: dict | None = None, period: dict | None = None) -> np.ndarray:
    """Whether each trigger has one at another station of the same pull
    within the P travel time between the two stations plus `slack_s`.

    Args:
        trig: columns `pull`, `station`, `t` (seconds), `lat`, `lon`.
        shifts: {(pull, station): seconds} to slide a station's triggers by,
            wrapped around `period[(pull, station)]` seconds from the window
            start `t0`; for the chance estimate.
    """
    t = trig["t"].to_numpy(float).copy()
    if shifts:
        for k, (pull, sta) in enumerate(zip(trig["pull"], trig["station"])):
            s = shifts.get((pull, sta), 0.0)
            if s:
                t0, length = period[(pull, sta)]
                t[k] = t0 + (t[k] - t0 + s) % length
    out = np.zeros(len(trig), bool)
    for _, labels in trig.groupby("pull").groups.items():
        idx = np.asarray(labels)
        if len(idx) < 2:
            continue
        sta = trig["station"].to_numpy()[idx]
        lat = trig["lat"].to_numpy(float)[idx]
        lon = trig["lon"].to_numpy(float)[idx]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if sta[a] == sta[b]:
                    continue
                d = float(haversine_km(lat[a], lon[a], lat[b], lon[b]))
                if abs(t[idx[a]] - t[idx[b]]) <= d / VP_KM_S + slack_s:
                    out[idx[a]] = out[idx[b]] = True
    return out


def main(argv=None):
    from onset.evaluate import load_model, predict
    p = argparse.ArgumentParser(prog="onset audit-noise", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--threshold", type=float, default=None,
                   help="Defaults to the run's validation operating point.")
    p.add_argument("--slack", type=float, default=2.0,
                   help="Seconds added to the inter-station P travel time.")
    p.add_argument("--slides", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg, dcfg, tcfg = load_model(a.run_dir, device, a.checkpoint)
    thr = a.threshold
    if thr is None:
        thr = json.loads((Path(a.run_dir) / "val_best.json").read_text())["summary"]["threshold"]
    # Whole noise windows at their own clock time: no lead-in splice, which
    # would shift them, and no second event.
    dcfg.eval_lead_in_s, dcfg.eval_second_every = 0.0, 0
    rule = metrics.TriggerRule.from_config(tcfg, mcfg)
    fs, s = mcfg.sample_rate, mcfg.stride
    period, hours_by_station = {}, {}

    def triggers(ds, audited):
        _, noise = predict(model, ds, device, workers=a.workers)
        rows = []
        for n in noise:
            r = ds.rows.iloc[n["index"]]
            t0 = pd.Timestamp(r.start_time).timestamp()
            tt = metrics.token_times(len(n["p"]), s, fs)
            if audited:
                hours_by_station[r.station] = (hours_by_station.get(r.station, 0.0)
                                               + (~n["missing_tokens"]).sum() * s / fs / 3600)
            period[(r.event_id, r.station)] = (t0, len(n["p"]) * s / fs)
            for e in metrics.trigger_tokens(n["p"], n["dt"], thr, thr * 0.5, s, fs, rule):
                rows.append({"pull": r.event_id, "network": r.network, "station": r.station,
                             "split": a.split if audited else r.split,
                             "lat": r.station_lat, "lon": r.station_lon, "t": t0 + tt[e],
                             "utc": str(pd.Timestamp(t0 + tt[e], unit="s")),
                             "p": float(n["p"][e]), "key": r.key, "audited": audited})
        return rows

    ds = OnsetDataset(a.data, a.split, dcfg, mcfg, False, ("noise",))
    rows = triggers(ds, True)
    hours = sum(hours_by_station.values())
    if not rows:
        print(f"{hours:.1f} h of {a.split} noise, no triggers at threshold {thr:.5f}")
        return
    # The same pulls at the stations of the other splits: the partners.
    pulls = {r["pull"] for r in rows}
    partners = 0
    for other in ("train", "val", "test"):
        if other == a.split:
            continue
        ods = OnsetDataset(a.data, other, dcfg, mcfg, False, ("noise",))
        ods.rows = ods.rows[ods.rows.event_id.isin(pulls)].reset_index(drop=True)
        partners += len(ods.rows)
        if len(ods.rows):
            rows += triggers(ods, False)
    trig = pd.DataFrame(rows)
    audited = trig["audited"].to_numpy()
    trig["coincident"] = coincidences(trig, a.slack)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(a.slides):
        shifts = {k: rng.uniform(0, v[1]) for k, v in period.items()}
        null.append(int(coincidences(trig, a.slack, shifts, period)[audited].sum()))
    obs = int(trig["coincident"][audited].sum())
    at = trig[audited]
    mu, sd = float(np.mean(null)), float(np.std(null))
    print(f"{a.run_dir} on {Path(a.data).name} {a.split} noise, threshold {thr:.5f}")
    print(f"  {hours:.1f} h, {len(at)} triggers ({len(at) / hours:.2f}/h) at "
          f"{at.station.nunique()} stations, in {at.pull.nunique()} pulls; those pulls "
          f"scored at {partners} windows of other-split stations as partners")
    rate = (at.station.value_counts().rename("triggers").to_frame()
            .assign(hours=lambda d: d.index.map(hours_by_station))
            .assign(per_hour=lambda d: d.triggers / d.hours))
    print("  by station\n" + rate.round(2).to_string().replace("\n", "\n    "))
    print(f"  coincident with another station: {obs} ({obs / len(at):.0%}); "
          f"by chance {mu:.1f} +- {sd:.1f}")
    if partners == 0:
        print("  -> no other station has windows of these pulls: the test cannot tell")
        return
    excess = obs - mu
    if excess > 3 * max(sd, 1.0):
        print(f"  -> about {excess:.0f} triggers look like earthquakes seen at several "
              "stations: the noise holds events the catalogue does not list")
    else:
        print("  -> no more coincidences than chance: the false triggers are the model's own")
    hit = trig.loc[trig.coincident & trig.audited, "pull"].unique()
    top = (trig[trig.coincident & trig.pull.isin(hit)].groupby("pull")
           .agg(stations=("station", "nunique"), first=("utc", "min"))
           .sort_values("stations", ascending=False).head(10))
    if len(top):
        print("\n  pulls with the most coincident stations\n" + top.to_string())
    out = Path(a.run_dir) / f"audit_noise_{a.split}.csv"
    trig.to_csv(out, index=False)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
