# onset: manual

How to build data, train, evaluate and ship the detector, and how to run and
compare it inside ayzek. The reasons behind the design are in `DESIGN.md`;
the data and its labels are in `DATA.md`.

## 0. Setup

```bash
cd ~/Projects/Codings/onset
uv sync --extra build          # obspy + scipy, for anything that reads miniSEED
uv run --extra build pytest    # 29 tests, about a minute
```

- **VIRTUAL_ENV warning.** If your shell has another project's venv active,
  `uv` prints a warning that `VIRTUAL_ENV` doesn't match. It's harmless; run
  `unset VIRTUAL_ENV` to silence it.
- **Where the data is found:**

| variable | default | used by |
|---|---|---|
| `ONSET_FDSN_ROOT` | `~/Projects/sismokaos/data_downloader` | `build-fdsn` (pulls, catalogue, stations) |
| `ONSET_STEAD_DIR` | `~/Projects/Codings/STEAD` | `build-stead`, `validate-aic` |

- **Command list:** run `uv run onset` with no arguments. Every command has
  `--help`.

## 1. The loop at a glance

```
build-fdsn ─→ datasets/fdsn_v1 ─→ train ─→ runs/<name> ─┬─→ evaluate    (test split, latency vs false triggers)
                                                        ├─→ replay      (continuous miniSEED, Python)
                                                        └─→ ayzek: export_transformer.py ─→ ayzek --detector transformer
                                                                                         └─→ compare_detectors.py (6 s vs transformer)
```

## 2. Data

```bash
uv run --extra build onset build-fdsn --out datasets/fdsn_v1 --workers 8     # ~2.5 min, 9 GB
```

**What it produces:**
- `waveforms.h5`: filtered traces.
- `index.csv`: one row per trace, with kind, split, station, P sample, label
  source and tolerance.
- `build.json`: the arguments, plus every dropped trace counted by reason.

**Useful flags:**
- `--limit N`: only the first N event files, for a quick smoke build.
- `--context-seconds`, `--noise-seconds`: default 120 each.
- `--max-missing`: default 0.10.

**A pull that starts before origin** (for example
`EARTHQUAKE_BATCHES = [("window_m60_p120", -60, 120)]` with
`SEARCH_RADIUS_DEG = 1.8`):

```bash
uv run --extra build onset build-fdsn --out datasets/fdsn_wide \
    --events-dir $ONSET_FDSN_ROOT/raw/data/batched_waveforms/window_m60_p120 \
    --event-start-offset -60 --event-seconds 180
```

- `--event-start-offset` is where the files start relative to origin.
- Every event trace gets distance, back-azimuth and component labels for the
  geometry head.
- Traces whose event would not be visible at that distance are dropped
  (`catalog.VISIBILITY`).

**Rebuilding:**
- A rebuild is only needed if the *stored* data changes: filter, window
  lengths, labels or contamination rules.
- Changes to crops, splices, gaps and so on live in the loader (`data.py`) and
  take effect on the next training run with no rebuild.

**STEAD, the fallback:**

```bash
uv run --extra build onset validate-aic --n 3000        # how good the AIC P labels are
uv run --extra build onset build-stead --out datasets/stead_v1 --max-traces 100000
```

STEAD only reads chunks whose `.csv` and `.hdf5` are both extracted. Use
`7z x chunkN.zip`, not unzip. `chunk5.zip` on disk is a broken download.

## 3. Train

```bash
uv run onset train --data datasets/fdsn_v1 --out runs/fdsn_v2
```

- **Time:** about 40 epochs × 1–2 min on the 3060 Ti.
- **Log:** one line per epoch. See §6 for how to read it.
- **Flags:** every field of `ModelConfig`, `DataConfig` and `TrainConfig` in
  `config.py` is a flag. For example:

  ```bash
  uv run onset train --data datasets/fdsn_v1 --out runs/try_small \
      --d-model 48 --n-layers 3 --window-tokens 60 \
      --lead-in-p 0.9 --ctx-drop 0.3 --fa-target-per-hour 0.5 --epochs 30
  ```
- **Adding STEAD:** `--fallback datasets/stead_v1 --fallback-weight 0.2`.
- **Geometry head (location):** `--geometry 1`, with `--geo-weight` for its
  share of the loss (default 0.1). `evaluate` then prints distance and
  back-azimuth error by time since P, and ayzek locates from it (§8).
- **Quick smoke run:** `--epochs 2 --steps-per-epoch 50`.

A run directory holds:

| file | what |
|---|---|
| `config.json` | the full resolved config; `evaluate`, `replay` and the ayzek export read it |
| `best.pt` | weights of the best epoch (recall within 1 s at the false-trigger budget) |
| `last.pt` | weights of the last epoch |
| `history.jsonl` | one JSON line per epoch: losses and the full validation summary |
| `val_best.json` | the best epoch's threshold sweep and operating point; the threshold ayzek uses comes from here |

Use a **new `--out` per run**. Re-using a directory appends to `history.jsonl`
and overwrites the weights.

## 4. Evaluate on held-out stations

```bash
uv run onset evaluate runs/fdsn_v2 --data datasets/fdsn_v1 --split test
uv run onset evaluate runs/fdsn_v2 --data datasets/fdsn_v1 --split test --no-context
```

- `--no-context` scores every trace as if the station had just come up, with
  no background summary yet.
- `--fa-target 0.5` re-picks the operating point for a different
  false-trigger budget.
- **Writes** `runs/<name>/eval_<dataset>_<split>.json` (the sweep) and `.csv`
  (one row per event: latency, early trigger, onset error, magnitude,
  distance).

**Validation and test traces get a 40 s seamless lead-in** of the station's
older noise, set by `DataConfig.eval_lead_in_s`. That's longer than the model
can look back, so the numbers describe a continuous stream, not a trace that
starts at origin. See DESIGN §4 for why that matters.

## 5. Replay continuous data (Python)

```bash
R=~/Projects/sismokaos/data_downloader
B=$R/raw/data/batched_waveforms
uv run --extra build onset replay runs/fdsn_v2 \
    $B/day_before_24h/event_543430_raw.mseed $B/window_post_60s/event_543430_raw.mseed \
    --out replays/543430 \
    --catalog $R/catalogs/catalog_current.csv --stations $R/catalogs/station_coords.csv
```

**Inputs:**
- All files are read as one recording. A day of lead-in plus its event window
  make one stream.
- Every station in the files is scored.
- `--catalog` and `--stations` add scoring against catalogued arrivals. They
  are optional.

**Outputs:**
- `<NET.STA>.npz`: every token's time, p and dt.
- `<NET.STA>_triggers.csv`: each trigger's time, p and dated onset, and whether
  it matched a catalogued arrival.
- `summary.json`: per station, the hours scored, triggers, context refreshes,
  detections and latency.

**Flags:**
- `--threshold`: overrides the run's validation operating point.
- `--quiet`: how low p must stay for a minute to refresh the station context
  (default 0.3).

Loading a replay for a quick look:

```python
import numpy as np
z = np.load("replays/543430/KO.GAZ.npz")
t, p, dt = z["t"], z["p"], z["dt"]      # seconds from the stream start, prob, seconds since onset
```

**The quickest regression test for the "data just began" shortcut** scores
the same event twice, alone and behind the 24 h before it:

```bash
E=360571
uv run --extra build onset replay runs/fdsn_v2 $B/window_post_60s/event_${E}_raw.mseed --out replays/${E}_alone
uv run --extra build onset replay runs/fdsn_v2 $B/day_before_24h/event_${E}_raw.mseed \
    $B/window_post_60s/event_${E}_raw.mseed --out replays/${E}_continuous
cat replays/${E}_alone/*_triggers.csv; tail -3 replays/${E}_continuous/*_triggers.csv
```

The event's origin is where the `window_post_60s` file starts. Compare the
first trigger after it in each run. A healthy model fires at nearly the same
moment both ways. The flawed v1 fired 2–14 s later with the 24 h in front
(DESIGN §4). The 15 events with a day of lead-in: 360571, 446014, 448444,
493263, 538720, 543429, 543430, 615784, 618388, 633881, 638244, 648158,
658148, 665471 and 687329.

## 6. Reading the numbers

**Training log line:**
```
epoch 5  loss 0.1141 (bce 0.1045 dt 0.096)  val recall@1s 0.462 @thr 0.99000 (0.71 FA/h)  lat p50 1.00s  second@2s 0.310  score 0.432
```
- `bce`: the loss for "is an event under way". 0.69 is a coin flip; it should
  fall.
- `dt`: the error of "seconds since onset", in seconds. It should fall.
- `val recall@1s`: the share of validation events caught within 1 s of P, at
  the threshold that keeps validation noise under the false-trigger budget.
- `second@2s`: the share of second onsets (an event added into another's coda,
  every 4th validation event) caught within 2 s. This is what the dt restart
  in ayzek's trigger depends on.
- `score`: recall within 1 s over first and second onsets together. It decides
  `best.pt`. Triggers are counted with ayzek's rule (rising edge or dt
  restart, 15 s apart between P dates).
- `@thr … (FA/h)`: that threshold, and its false triggers per hour. The
  threshold is not a confidence: training is balanced, so what matters is only
  where it lands on real noise.

**Evaluate summary:**
- `recall ≤0.25s … ≤4s`: the latency curve, meaning the share of events caught
  within each delay. A better model moves the whole row up.
- `ever`: caught at any time. If "ever" is high and "≤1s" low, the model is
  firing on S or the coda, not on P.
- `early (pre-P) triggers`: firings before P on event traces. They count as
  false alarms.
- `median onset error from dt`: how well "now − dt" locates P at the moment of
  triggering. This is what places the picker's window.
- The per-magnitude and per-label-source tables show where recall comes from.

## 7. Ship to ayzek

ayzek lives at `~/Projects/Codings/ayzek/ayzek_code`. The transformer work is
on branch `transformer-detector`.

```bash
cd ~/Projects/Codings/ayzek/ayzek_code
ninja -C build-release                                       # build
uv run --project tools python tools/export_transformer.py --run ../../onset/runs/fdsn_v2
meson test -C build-release test_transformer                 # C++ vs PyTorch on DEMI
```

**What the export writes:**
- `models/transformer.ayzw`: the weights, geometry, filter coefficients and the
  operating threshold from `val_best.json`. `--threshold X` overrides the
  threshold.
- `data/fixtures/transformer.ayzw`: reference outputs for the test.

**A run trained with `--geometry 1`** exports its geometry head too, and the
fixtures include its outputs, which `test_transformer` compares at every
token. The export then prints the head's distance and back-azimuth on DEMI
next to the true ones for the Sındırgı M4.9. `meson test -C build-release
test_locate` checks ayzek's locator against the scenarios of
`tests/test_locate.py`.

**Exports overwrite each other.** To keep several models side by side, copy
the `.ayzw` somewhere else and point ayzek at it with `--transformer FILE`.

The test compares three stages:
- the causal filter across a gap;
- the network with and without a station context;
- raw counts all the way to token outputs.

Expect errors around 1e-7. A failure after a code change in `onset/model.py`
means the C++ port in `src/transformer.cpp` needs the same change.

## 8. Run ayzek with either detector

```bash
build-release/app/ayzek --speed 0 --detector 6s          --catalog tests/catalogs/demo_ko.csv data/demo_ko/*.mseed
build-release/app/ayzek --speed 0 --detector transformer --catalog tests/catalogs/demo_ko.csv data/demo_ko/*.mseed
```

| flag | meaning |
|---|---|
| `--detector 6s` | the 3-seed 6 s window detector (the default; `model` still works) |
| `--detector transformer` | the streaming transformer, one output per 0.1 s |
| `--detector stalta` | the classical STA/LTA reference |
| `--transformer FILE` | transformer weights (default `models/transformer.ayzw`) |
| `--locate geometry` | locate from the geometry head, no picker (the default when the model has the head) |
| `--locate picks` | locate from P and S picks (the default otherwise) |
| `--threshold P`, `--release P` | transformer: default to the model's operating point and half of it |
| `--scores DIR` | per-station CSV of every window's (6 s) or token's (transformer) probability |
| `--record FILE` | every detection, pick and magnitude, for offline analysis |
| `--speed 0` | as fast as possible; `1` is real time |

**Differences between the two detectors:**
- **P time.** The transformer dates P itself (trigger time − dt), so the
  STA/LTA anchor is off for it.
- **Gaps.** It runs through gaps, and its gap channel marks the missing
  samples. The 6 s detector drops every window that touches a gap.
- **Report table.** Under `--detector transformer`, the "windows" column in the
  station table counts 0.1 s tokens.

## 9. Compare 6 s against the transformer

```bash
cd ~/Projects/Codings/ayzek/ayzek_code
uv run --project tools python tools/compare_detectors.py --only demo_ko,marmara_ko,quiet_ko --jobs 6
uv run --project tools python tools/compare_detectors.py --jobs 6 --workdir cmp/ --json compare.json   # all 10 datasets
uv run --project tools python tools/compare_detectors.py --workdir cmp/ --reuse                        # rescore without replaying
```

Every scorecard dataset is replayed twice, identically except for
`--detector`.

- **Datasets:** the public KO sets (`demo_ko`, `marmara_ko`, `quiet_ko`) and
  the AFAD sets (`demo`, `marmara`, `sindirgi61`, `noise_*`).
- **Station level, the detector alone:**
  - recall within 0.5 / 1 / 2 / 4 / 8 s of P, and median latency;
  - P-dating error;
  - unmatched triggers per station-hour;
  - recall by epicentral distance.

  The reference P is TauP, refined by an AIC pick on each station's own
  vertical.
- **Network level, the whole pipeline:** catalogue events detected, unmatched
  alarms, alarm delay, magnitude error, and false alarms per day on the quiet
  sets.
- **Cost:** detector milliseconds per station-hour.

**Tips:**
- `--jobs 6` runs six replays at once, about 2 cores each. Don't run it while
  training if you care about training speed.
- `--workdir` keeps each replay's stdout and `--record` file, so `--reuse` can
  rescore after a change to the analysis without replaying.
- Delete a dataset's two files from `--workdir` to force it to replay, for
  example after exporting a new model.
- `--args '...'` passes extra ayzek flags to both runs, for example
  `--args '--min-stations 3'`.

**The scorecard** (`tools/scorecard.py --args '--detector transformer'`) also
works on one detector at a time, and `--compare A.json B.json` puts two
scorecards side by side.

## 10. Typical recipes

**A training change, end to end:**
```bash
cd ~/Projects/Codings/onset
uv run onset train --data datasets/fdsn_v1 --out runs/<new>
uv run onset evaluate runs/<new> --data datasets/fdsn_v1 --split test
cd ~/Projects/Codings/ayzek/ayzek_code
uv run --project tools python tools/export_transformer.py --run ../../onset/runs/<new>
meson test -C build-release test_transformer
rm -f cmp/*_transformer.*        # keep the 6 s runs, replay only the transformer
uv run --project tools python tools/compare_detectors.py --jobs 6 --workdir cmp/ --reuse --json compare_<new>.json
```

`--reuse` only reuses runs whose files are present, so the deleted transformer
runs replay and the 6 s ones don't.

**Is a new model better?** In order of trust:
1. `compare_detectors.py`: real continuous replays, same input as the 6 s
   detector.
2. `onset replay` on the `day_before_24h` + `window_post_60s` streams.
3. `onset evaluate` on the test split.

Validation numbers pick checkpoints; they are not the verdict.

**Change the architecture:**
- Python side: edit `ModelConfig`, or `model.py` for new layers.
- If `model.py` changed, port the change to `ayzek/src/transformer.cpp`. The
  export and the fixture test will fail until the two agree.
