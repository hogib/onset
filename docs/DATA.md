# onset: data

FDSN pulls from KOERI (network KO) are the primary source. STEAD is the
fallback. Both are written in one store format (`store.py`), so training reads
them the same way.

## Sources on disk

All FDSN paths are relative to `ONSET_FDSN_ROOT`, which defaults to the
`data_downloader` checkout. STEAD is read from `ONSET_STEAD_DIR`.

| pull | span relative to catalogue origin | role |
|---|---|---|
| `raw/data/batched_waveforms/window_post_60s` | 0 … +60 s | event traces |
| `raw/data/batched_noise_waveforms/noise_pre_3h` | −3 h, 5 min | station context |
| `raw/data/batched_noise_waveforms/noise_pre_6h` | −6 h, 5 min | noise traces |
| `raw/data/batched_waveforms/day_before_24h` + `window_post_60s` | −24 h … +60 s, 15 events | continuous replay tests |

- One file can hold several stations, and each station becomes its own trace.
- About a third of the event files have context and noise pulls. Every event
  file has an event trace.
- Catalogue: `catalogs/catalog_current.csv` (AFAD). Stations:
  `catalogs/station_coords.csv`.

## Building `datasets/fdsn_v1`

```bash
uv run --extra build onset build-fdsn --out datasets/fdsn_v1 --workers 8   # ~2.5 min
```

Per station:
1. **Instrument:** one 3-component instrument, preferring HH over BH over EH,
   and a blank location code first.
2. **Grid:** put on a 100 Hz grid, with gaps kept as masks rather than
   interpolated.
3. **Filter:** causally, as described in DESIGN §3.
4. **Label P:**
   - predict P and S with TauP (iasp91) from the catalogue hypocentre;
   - refine P with an AIC pick on the vertical, searching −2 … +3 s around the
     prediction, stopping 0.5 s before the predicted S, and never inside the
     first 1.5 s (filter start-up);
   - accept the pick if the second after it has twice the RMS of the second
     before it.
5. **Contamination:** drop a trace if a *visible* catalogued event arrives
   before its P. A context or noise window is dropped if any visible arrival
   falls inside it, or if one arrived before it recently enough that its coda
   still overlaps: coda duration from Md = 2 log10(τ) − 0.87, about 25 s at
   M2, 85 s at M3, 270 s at M4, 860 s at M5, capped at 1 h (`--coda-max-s`).
   The v1 build had only the inside check; its numbers below are from that.
   Noise windows (not context) are also dropped where the catalogue is
   likely incomplete: in a busy place and time, in the aftermath of an
   M ≥ 5 nearby, or where `onset audit-noise` found a multi-station event
   (MANUAL §2). "Visible" means within 50 km at any magnitude, 150 km at
   M2+, 400 km at M3+ or 1500 km at M4.5+ (`catalog.VISIBILITY`). The rule is
   generous on purpose: a lost clean window costs little, a kept contaminated
   one teaches the model that earthquakes are noise.
6. **Split:** by station, from a hash of `NET.STA`, 80 / 10 / 10. Every rebuild
   and every source puts a station in the same split.

**Result** (`build.json` has every drop, by reason):

| | train | val | test |
|---|---:|---:|---:|
| event traces | 44,522 | 3,815 | 6,136 |
| noise traces (120 s) | 16,011 (534 h) | 1,268 (42 h) | 2,123 (71 h) |
| station contexts (120 s) | 15,807 | 1,234 | 2,100 |

- The event traces cover 32,372 events at 181 stations. 35% of them have a
  station context.
- Magnitude: median M2.3, 95th percentile M3.3. 832 traces are M4 or above and
  69 are M5 or above.
- Distance: median 39 km, 95th percentile 54 km. The pulls were requested
  within 0.5°.
- P lies a median 7.6 s after the trace start (1st–99th percentile 1.7–11.3 s).
- Drops: 1,812 traces for gaps, 929 for an earlier arrival inside the window.

## Label quality, measured

**TauP alone is not good enough for onset timing.** On the KO pulls the
accepted AIC picks land a median **+0.86 s** after the iasp91 prediction
(5th–95th percentile −0.35 … +1.93 s). iasp91 is fast for this crust. A
detector trained on bare TauP labels would learn to fire a second "early" by
the labels' reckoning, and its latency numbers would be off by that much.

**The AIC refinement was checked against manual picks** using STEAD, the only
ground truth on disk (`onset validate-aic`). Each manual pick was hidden behind
a TauP-like error, N(−0.9 s, 0.6 s), and then refined:

| | median error | 68% within | 90% within | 95% within |
|---|---:|---:|---:|---:|
| prediction alone | −0.93 s | 1.22 s | 1.67 s | 1.88 s |
| **AIC, accepted picks** (96% of traces) | **+0.02 s** | **0.03 s** | **0.13 s** | **0.23 s** |

Of the accepted picks, 87.6% are within 0.1 s, 96.7% within 0.3 s and 99.6%
within 1.0 s.

So each trace carries `p_tolerance_s`, and tokens that close to P are left out
of the loss:

| `p_source` | tolerance | share of FDSN events |
|---|---:|---:|
| `aic` | 0.3 s | 85% |
| `taup` (AIC rejected) | 1.0 s | 15% |
| `manual` (STEAD) | 0.1 s | — |

**Caveat.** The 0.3 s comes from STEAD, where SNR, instruments and
preprocessing differ from KO. It is an estimate for KO, not a measurement on
KO.

## Onsets in an event trace's own coda (`later_p`)

An event window runs 60 s or more past origin, and in a sequence it often
holds another catalogued event after its own P. The build drops a trace with
a catalogued arrival *before* its P, but one after it used to stay in
unlabelled, so the model was taught that a real onset inside a coda is more
coda: dt kept counting through it, the opposite of the restart ayzek's
trigger fires on.

Each such arrival is now labelled (`onset/later.py`; `build-fdsn` writes it,
`onset label-later` adds it to an existing store): predicted with TauP and
refined by AIC like the first P. An accepted pick is a labelled onset, with
a tolerance of 0.5 s, and dt restarts there as for a synthetic second event;
otherwise only the TauP prediction is kept, and dt is not trained from 1 s
before it to 10 s after (plus 1 s of tolerance each side), where it depends
on where the onset really was.

| store | event traces with one | arrivals picked | not picked | within 39 s of P (crop reach) |
|---|---:|---:|---:|---:|
| `fdsn_wide_x` | 6744 (5.6%) | 4787 | 2395 | 1130 picked, 894 not |
| `fdsn_v1` | 1176 (2.2%) | 780 | 438 | 475 picked, 278 not |

On 400 `fdsn_wide_x` traces, the accepted picks land a median +0.79 s after
the TauP prediction (5th–95th percentile −0.74 … +2.37 s), as the first-P
picks do (+0.86 s), so they are mostly the arrival and not a burst of coda;
about 13% lie more than a second from that peak. Their median SNR (the
second after the pick over the second before) is 4.3.

The picked ones in the validation split (499 on `fdsn_wide_x`) are scored on
their own as real coda onsets ("catalogued onsets in the coda" in
`evaluate`), next to the synthetic second events. Traces that hold one are
not used as sources for synthetic second events, whose own coda onsets would
come along unlabelled. The catalogue misses small events, so this labels the
catalogued part of the problem only.

## STEAD (fallback)

```bash
uv run --extra build onset validate-aic --n 3000
uv run --extra build onset build-stead --out datasets/stead_v1 --max-traces 100000
uv run onset train --data datasets/fdsn_v1 --fallback datasets/stead_v1 --fallback-weight 0.3 ...
```

**What is on disk:**
- `chunk2` is extracted (csv + hdf5).
- `chunk3` and `chunk4` are still zipped. `7z x` handles them; unzip does not,
  because the archives overflow 32-bit headers.
- `chunk5.zip` is a 2.4 KB Google Drive "virus scan warning" HTML page, not
  data. It needs a fresh download.
- `chunk1` (STEAD's noise) is absent. STEAD therefore contributes events only.

**What the build keeps:** earthquake traces with a manual P, and all three
components alive.
- Traces are reordered from STEAD's E N Z to Z N E and filtered with the same
  causal band-pass.
- `station_split` uses the same hash as FDSN.

**Why STEAD is only a fallback:**
- It holds US stations, not Turkish ones.
- It has no station context and no lead-in, so each P has at most ~10 s of
  history.
- Its preprocessing history is unverified (whether its band-pass was
  zero-phase).
- Its events are small: median M1.2.

## Continuous replay sets

```bash
R=$ONSET_FDSN_ROOT/raw/data/batched_waveforms
uv run --extra build onset replay runs/<run> \
    $R/day_before_24h/event_543430_raw.mseed $R/window_post_60s/event_543430_raw.mseed \
    --out replays/543430 \
    --catalog $ONSET_FDSN_ROOT/catalogs/catalog_current.csv \
    --stations $ONSET_FDSN_ROOT/catalogs/station_coords.csv
```

Each of the 15 `day_before_24h` files ends at an event's origin time, and its
`window_post_60s` file continues from there. The two together make one
continuous recording.

`event_543430` covers GAZ and KMRS from 2023-02-05 01:36 to 2023-02-06 01:37
UTC. That includes the Mw 7.8 Kahramanmaraş mainshock at 01:17 and the M5.6
at 01:26.

These streams are gappy: one multi-station file has 35% vertical coverage. That
is expected. The gaps are what the mask channel and the gap augmentation exist
for.
