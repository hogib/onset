# onset: manual

How to build data, train, evaluate and ship the detector, and how to run and
compare it inside ayzek. The reasons behind the design are in `DESIGN.md`;
the data and its labels are in `DATA.md`.

## 0. Setup

```bash
cd ~/Projects/Codings/onset
uv sync --extra build          # obspy + scipy, for anything that reads miniSEED
uv run --extra build pytest    # 46 tests, a few seconds (minutes on a cold first run)
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

**A pull that starts before origin, over a wider radius.** In the
downloader (seismic_cli `src/download.py`):

```python
EARTHQUAKE_BATCHES = [("window_m60_p120", -60, 120)]
NOISE_BATCHES = [("noise_pre_3h", -11400, -10800),   # 10 min each; the default is 5
                 ("noise_pre_6h", -22200, -21600)]
SEARCH_RADIUS_DEG = 1.8
BASE_OUTPUT_DIR = Path("data_wide")        # a fresh tree: see below
NOISE_CONTAMINATION_BUFFER_SEC = 0         # the builder checks contamination itself
```

Then build from the same tree, event and noise folders alike:

```bash
R=$ONSET_FDSN_ROOT
uv run --extra build onset build-fdsn --out datasets/fdsn_wide \
    --events-dir  $R/data_wide/batched_waveforms/window_m60_p120 \
    --context-dir $R/data_wide/batched_noise_waveforms/noise_pre_3h \
    --noise-dir   $R/data_wide/batched_noise_waveforms/noise_pre_6h \
    --event-start-offset -60 --event-seconds 180 \
    --context-seconds 590 --noise-seconds 590
```

Three things decide whether the event traces get noise and context, which
the lead-in splice, the station context and validation all need. The first
wide build got it for 460 of 18,762 traces:

- **The same tree.** `--context-dir` and `--noise-dir` default to the v1
  pulls under `raw/data`. Left at that, only events and stations the v1 pull
  also had find noise.
- **A fresh output tree.** The downloader skips any file that exists. Noise
  batch names do not change with the radius, so noise saved by a 0.5° run for
  the same event keeps only its near stations, and every farther station is
  `noise_absent`.
- **The downloader's contamination buffer at 0.** It drops a noise window if
  any catalogued event anywhere in Türkiye falls within the buffer, whatever
  its distance: at the catalogue's rates that is half to nine tenths of all
  windows (more in a sequence). The builder's own check is by distance and
  magnitude, and also looks back for coda (below).

`--context-seconds` and `--noise-seconds` cannot exceed the pulled window,
less a few seconds: 290 for the default 5 min windows, 590 for 10 min ones. A
longer window gives training crops more variety and validation more noise
hours. `build.json` counts every trace dropped as `*_absent` or `*_short`.

- `--event-start-offset` is where the files start relative to origin.
- Every event trace gets distance, back-azimuth and component labels. The
  geometry head trains on the distance.
- Traces whose event would not be visible at that distance are dropped
  (`catalog.VISIBILITY`).
- A noise or context window is dropped if a visible catalogued event arrives
  inside it, or arrived early enough that its coda is still ringing: about
  25 s after an M2, 85 s after an M3, 270 s after an M4 and 860 s after an M5
  (`catalog.coda_seconds`, capped by `--coda-max-s`, default 3600; 0 checks
  inside the window only, as before).

**Noise the catalogue cannot vouch for.** The catalogue check only catches
listed events, and a catalogue misses small ones: most of all right after a
large event, and wherever a sequence is running. On the first wide build 27
multi-station events sat in the validation noise, none catalogued, most in
the hours and days after the 27 Oct 2025 Sındırgı M6.1; they set the
operating threshold at 0.994 and made recall swing from epoch to epoch. So
the builder also drops a *noise* window (not context) when:

| rule | flags (default) | drop reason |
|---|---|---|
| more than N catalogued events within R km in the H hours around it | `--max-active-events 3 --active-radius-km 75 --active-hours 12` (-1: off) | `noise_active` |
| it falls in the aftermath of an M ≥ 5 within 150 km: 30 days at M5, ×3.2 per magnitude unit | `--aftermath-mag 5 --aftermath-radius-km 150 --aftermath-days 30` (0: off) | `noise_aftermath` |
| `onset audit-noise` found it in a multi-station coincidence, or its pull had one spanning 3+ stations | `--exclude-noise runs/x/audit_noise_*.csv` | `noise_excluded` |

On that audit the first two rules alone would have removed 144 of 154
validation false triggers and 30 of the 31 in multi-station coincidences.
To catch the rest, audit a trained model on every split and rebuild with the
results:

```bash
for s in train val test; do uv run onset audit-noise runs/x --data datasets/fdsn_wide --split $s; done
uv run --extra build onset build-fdsn ... --exclude-noise runs/x/audit_noise_{train,val,test}.csv
```

**Onsets in the coda** (`later_p`, docs/DATA.md): `build-fdsn` labels the
catalogued arrivals after each event trace's own P. A store built before that
gets them in place, without a rebuild (the old index is kept as
`index.csv.bak`):

```bash
uv run --extra build onset label-later --data datasets/fdsn_wide_x
```

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

- **Before the first epoch** it prints the noise traces and hours per split.
  It stops if there is no training or validation noise (the model would learn
  that everything is an event, and the operating point would be undefined),
  and warns when validation noise is under 5 h (one false trigger then moves
  the operating point) or when under 80% of event traces have noise from
  their own station (no lead-in splice or context for the rest). Fix those in
  the build (§2), not in training.
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
  share of the loss (default 0.1) and `--geo-min-sd` for the smallest
  distance uncertainty it may state (default 0.1, about 10%; without a floor
  it keeps shrinking its uncertainty on training data and turns
  overconfident). `evaluate` then prints distance error by time since
  P, and ayzek locates from it (§8).
- **Quick smoke run:** `--epochs 2 --steps-per-epoch 50`.

A run directory holds:

| file | what |
|---|---|
| `config.json` | the full resolved config; `evaluate`, `replay` and the ayzek export read it |
| `best.pt` | weights of the best epoch (recall within 1 s at the false-trigger budget) |
| `last.pt` | weights of the last epoch |
| `history.jsonl` | one JSON line per epoch: losses and the full validation summary |
| `val_best.json` | the best epoch's threshold sweep and operating point; the threshold ayzek uses comes from here |
| `state.pt` | weights, optimiser, step and best score after the last completed epoch, for `--resume` |

**One run per directory.** `onset train` refuses an `--out` that already holds
a run, so the history, checkpoints and evaluations in a directory always
belong to one training. To continue an interrupted run (Ctrl-C, a crash, a
reboot), give the same `--data` and `--out` with `--resume`: it picks up after
the last completed epoch with the saved config, optimiser and learning-rate
schedule, and the other flags are ignored. `--overwrite` moves an old run's
files into `--out/previous_<time>/` and starts a new one there.

**Let a run finish.** The learning rate decays on a cosine over all
`--epochs`; the last epochs, at a low rate, usually give the best
checkpoint. A run stopped early exports a model that never had them;
`--resume` it instead of starting over.

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
- `<NET.STA>.npz`: every token's time, p and dt; with the geometry head also
  `dist_km` and `log_dist_sd`.
- `<NET.STA>_triggers.csv`: each trigger's time, p and dated onset, and whether
  it matched a catalogued arrival; with the geometry head, its distance at
  the trigger and 10 s later. Triggers follow ayzek's rule
  (rising edge or dt restart, 5 s apart).
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
epoch 5  loss 0.1141 (bce 0.1045 dt 0.096)  val recall@1s 0.462 @thr 0.99000 (0.71 FA/h)  lat p50 1.00s  second@2s 0.310 (dt min 2.6s)  score 0.432
```
- `bce`: the loss for "is an event under way". 0.69 is a coin flip; it should
  fall.
- `dt`: the error of "seconds since onset", in seconds. It should fall.
- `val recall@1s`: the share of validation events caught within 1 s of P, at
  the threshold that keeps validation noise under the false-trigger budget.
- `second@2s`: the share of second onsets (an event added into another's coda,
  every 4th validation event) caught within 2 s. This is what the dt restart
  in ayzek's trigger depends on.
- `dt min`: the median of the lowest dt within 3 s of each second onset's P.
  ayzek's restart fires only once dt is at or below the restart level
  (`--dt-reset-below`, 1 s by default, exported with the model), so this
  should fall below it; above it the model sees the new onset but the
  trigger misses it.
- `score`: recall within 1 s over first and second onsets together. It decides
  `best.pt`. Triggers are counted with ayzek's rule (rising edge or dt
  restart, 5 s apart between P dates).
- `@thr … (FA/h)`: that threshold, and its false triggers per hour. It is
  the lowest threshold within the budget, found by bisection between grid
  steps: outputs crowd against 1, where one grid step moves recall by tens of
  points. The
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
- `second onsets in the coda`: recall on the validation or test events that
  carry a second event in their coda (every 4th), and over both onsets. This
  is what catches aftershocks in ayzek. The line under it gives the median
  lowest dt near their P (ayzek's restart needs it at or below the restart
  level, 1 s by default).
- `how far is it (geometry head)`: distance error by time since P, and `within 1 sd`, the share inside the model's own uncertainty (0.68 when
  it is honest).
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
token. The export then prints the head's distance on DEMI next to the true
one for the Sındırgı M4.9. `meson test -C build-release
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
| `--dt-reset BELOW,FROM`, `--no-dt-reset` | transformer: also trigger when dt restarts while p stays high (default: the level exported with the model, else 2,5), or rising edges only |
| `--pick-anywhere` | transformer: let the picker search its whole 60 s window instead of near the transformer's P and before the next trigger |
| `--geo-sd-scale`, `--geo-max-z`, `--geo-max-err-km` | geometry locator settings, all off by default (ayzek `docs/impl/15-geometry-location.md`) |
| `--assess`, `--assess-csv FILE` | diagnostic: judge each alarm earthquake / possible / misfire / unclassified from S-P (ayzek `docs/impl/16-alarm-assessment.md`) |
| `--threshold P`, `--release P` | transformer: default to the model's operating point and half of it |
| `--scores DIR` | per-station CSV of every window's (6 s) or token's (transformer) probability |
| `--record FILE` | every detection, pick and magnitude, for offline analysis |
| `--speed 0` | as fast as possible; `1` is real time |

**Differences between the two detectors:**
- **P time.** The transformer dates P itself (trigger time − dt), so the
  STA/LTA anchor is off for it, and the picker looks for P near that P.
- **Trigger.** A rising edge, or a dt restart while p stays high, which is
  how an aftershock inside another event's coda is caught. A detection from
  a restart is not absorbed as coda by the network stage.
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
- **Location:** the locator each run used, events located, time from alarm
  to first location, epicentre error, and the error on the events both runs
  located.
- **Cost:** detector milliseconds per station-hour.

With `--args '--assess'` the network table adds the unmatched alarms the
assessment judges real ("unm. real").

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
