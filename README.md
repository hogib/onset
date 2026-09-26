# onset

A streaming earthquake detector. It is a causal transformer that scores every
0.1 s of a three-component station stream. It looks back over the last ~30 s
of signal and at a summary of the station's own background. Its goal is to
fire as soon as possible after the P arrival and to say when that arrival was,
so a phase picker gets the right window without delay.

- `docs/DESIGN.md`: the architecture and the reasons behind it.
- `docs/DATA.md`: the sources, the labels, and how good the labels are.
- `docs/MANUAL.md`: how to use the tooling here and in ayzek, command by command.

It is built to be transcribed into ayzek (`../ayzek/ayzek_code`, C++23,
hand-written inference). `stream.StreamingDetector` is the reference algorithm
for that port, and the tests hold it equal to the trained forward pass.

## Setup

```bash
uv sync --extra build      # the build extra (obspy, scipy) reads miniSEED; training needs only torch
uv run pytest
```

## Pipeline

```bash
# 1. data: FDSN pulls -> datasets/fdsn_v1 (~2.5 min, 9 GB)
uv run --extra build onset build-fdsn --out datasets/fdsn_v1

# 2. train (selects on recall within 1 s of P at <= 1 false trigger/h on val noise)
uv run onset train --data datasets/fdsn_v1 --out runs/fdsn_v1

# 3. latency and false triggers on the test stations
uv run onset evaluate runs/fdsn_v1 --data datasets/fdsn_v1 --split test
uv run onset evaluate runs/fdsn_v1 --data datasets/fdsn_v1 --split test --no-context

# 4. a day of continuous data, context refreshed online, scored against the catalogue
uv run --extra build onset replay runs/fdsn_v1 day.mseed event.mseed --out replays/x \
    --catalog .../catalog_current.csv --stations .../station_coords.csv
```

`onset` lists the commands, and each has its own `--help`. Every field of
`ModelConfig`, `DataConfig` and `TrainConfig` is a training flag, for example
`--window-tokens 60 --ctx-drop 0.3 --fa-target-per-hour 0.5`. Each run writes
`config.json` beside its weights, and `evaluate` and `replay` read it back
instead of taking the geometry as flags.

STEAD is a fallback source, `onset build-stead` then `onset train --fallback`.
See `docs/DATA.md`.

## Layout

| module | holds |
|---|---|
| `model.py` | `OnsetDetector`: causal stem, sliding-window ALiBi attention, context encoder, p/dt heads |
| `stream.py` | `StreamingDetector` (one token at a time, KV cache) and `score_blocks` |
| `config.py` | `ModelConfig`, `DataConfig`, `TrainConfig`, run config save/load |
| `conditioning.py`, `dsp.py` | asinh(x / σ_station) with a gap channel; causal band-pass |
| `labels.py` | per-token targets; AIC refinement of a predicted P |
| `catalog.py` | AFAD catalogue, stations, TauP, visible-arrival checks |
| `store.py` | the dataset format: `waveforms.h5` + `index.csv` |
| `build_fdsn.py`, `build_stead.py` | the two sources -> stores; `validate-aic` |
| `data.py` | crops, lead-in, gaps, context -> examples |
| `metrics.py` | latency, early triggers, false triggers per hour, threshold sweep |
| `train.py`, `evaluate.py`, `replay.py`, `cli.py` | the commands |
