# onset: design

A detector for a continuous seismic stream. It fires as soon as possible after a
P arrival, so the phase picker gets the right window as soon as possible. It is
the first of three stages: detector, then a noise/P/S picker, then a
magnitude regressor. Only the detector is built here; ayzek supplies the
other two. With the geometry head (`ModelConfig.geometry`) the detector also
says where the event is, which ayzek uses to locate events without the
picker (§6).

The deployment target is ayzek (C++23, hand-written inference, Raspberry Pi).
Everything below is chosen so that the model can be transcribed there and
checked against PyTorch sample by sample.

---

## 1. What was wrong with the previous detector

The cascade_impl detector, deployed in ayzek, scores fixed 6 s windows every
0.5 s. Each window is scored from scratch.

| property | consequence |
|---|---|
| Every window is scored from scratch | The window is its own reference for "normal". There is no LTA, so it cannot tell "louder than this station usually is" from "loud". |
| Trained with P at 3.5 s into the window | It waits for about 3.5 s of post-P signal before it is confident. |
| Label smoothing to 0.1 / 0.9 | Outputs saturate at 0.90–0.907. A threshold on that plateau decided by the third decimal whether an Mw 6.2 triggered (KIRK, 0.8992). |
| Trigger rule of 8 windows ≥ 0.8 | A hand-built stand-in for temporal context, costing 3.5 s. |
| Judged on AUC | It never measured the one thing that matters: how soon after P it fires. |

## 2. The model

```
x (B, T, 4)  Z N E + gap mask, 100 Hz
  └─ causal conv stem   k7 s2 → k5 s1 → k5 s5     one token per 0.1 s (10 samples)
       └─ 4 × block:
            sliding-window causal self-attention   80 tokens = 8 s per layer, ALiBi
            cross-attention to the station context
            feed-forward
       └─ per token:  p   "an event is under way"   (sigmoid)
                      dt  seconds since its onset     (0–10 s)

ctx (B, Tc, 4)   60 s of the station's quiet signal
  └─ same stem ─ 8 learned queries cross-attend ─ 1 self-attention layer ─→ 8 tokens
```

397k parameters (`ModelConfig` defaults).

### Causality is exact, down to the sample

- The stem's convolutions are padded on the left by `kernel − stride`, not
  `kernel − 1`. Token *j* then ends exactly on sample `10j + 9`. The usual
  `k − 1` padding would leave each output blind to its newest `s − 1` samples,
  which adds latency for nothing.
- Normalization in the stem is a LayerNorm over channels at each time step. It
  is not BatchNorm or GroupNorm, which mix time steps.
- Self-attention is masked to the past 80 tokens, with ALiBi distance
  penalties. With 4 heads the slopes run from 1/4 to 1/256 per token, so one
  head is nearly local and another weighs the whole window evenly. There are no
  absolute positions, which is correct for a stream that has no beginning.
- Stacking 4 layers of 8 s windows gives an effective lookback of 31.7 s
  (`ModelConfig.lookback_tokens`).

`tests/test_model.py` holds the model to this:
- a sample in the future never changes a past output;
- token *j* responds to sample `10j + 9`;
- samples older than the lookback have no effect;
- the one-token-at-a-time `StreamingDetector` reproduces the full forward pass
  to 1e-5.

### Two timescales: why the lookback is split

The design question was how to use the fact that a stream can be looked back
on. The answer is shaped by the data:
- STEAD has P 4–10 s into each trace (median 7 s).
- The FDSN event pulls start at origin time, which is about 7 s before P for
  events within 55 km.

Nothing available has long *contiguous* history before P at training scale.
So the lookback is split into two parts:

- **Near past, contiguous (8 s per layer, ~32 s stacked).** This is where
  onsets are recognized. The data covers it, and the lead-in augmentation (§4)
  covers the part beyond 15 s.
- **Station context, not contiguous (60 s of quiet signal).** This answers
  "what does this station normally look like": microseism level, cultural
  noise, which components are noisy. That doesn't have to be adjacent to now.
  - In training it comes from the same station 3 h before the event (the
    `noise_pre_3h` pull).
  - In deployment it is refreshed every minute from quiet data, the way ayzek's
    `NoiseBaseline` already works, so it will be fresher than in training.

**This is also the practical form of "continuous learning".** The weights stay
frozen, and the model adapts to each station in context, from its own recent
signal:
- there is no online backpropagation on a Pi;
- there are no labels to learn from in the field, and nothing to steer it;
- it cannot drift toward treating an aftershock sequence as normal;
- a replay reproduces a run exactly.

Without context the model uses a learned null context. That case is real: a
station that has just started, or one whose background is too gapped to trust.
It is trained 20% of the time (`ctx_drop`).

### Two heads: p and dt

`p` is "an event is under way". It switches on at P and stays on through the
coda. `dt` is how long ago it started. Together they give the trigger logic
something a single probability cannot:

- **A new onset** has high p and small dt. Trigger, and hand the picker a
  window starting at `t − dt`.
- **The coda of an old event** has high p and large dt. Don't re-trigger. ayzek
  currently handles this with a 40 s coda absorption at the network level.

Evaluation reports how well `t − dt` at the trigger locates P
(`onset_abs_err_p50_s`). That estimate is what lets the detector place the
picker's window instead of the "P is 3.5 s into the window" convention.

## 3. Input conditioning

```
counts ─ causal Butterworth 1–45 Hz, 4th order, restarts after gaps ─ ÷ σ_station ─ asinh
```

- **The filter is causal** (`dsp.py`). A zero-phase filter smears onset energy
  ahead of the arrival, so a model trained on it learns to "see" P early, and
  deployment, which can only filter forward, takes that ability away.
- Each contiguous run is filtered on its own, from steady-state initial
  conditions, as a real-time filter has to restart after a gap.
- **σ_station** is the per-component RMS of the station context. Without
  context it is the RMS of the stream's first second. Dividing by it cancels
  instrument gain and leaves "how far above this station's background", which
  is the quantity a detector should trigger on.
- **asinh** keeps a near-field M6 inside bf16/fp16 range while leaving noise
  resolved. The previous project lost whole runs to fp16 overflow on raw
  σ-multiples (3.6e5 against a limit of 65504).
- **Gaps** are 0 on Z/N/E and 1 on a fourth channel, so the model knows a
  missing sample from a quiet one. Training inserts synthetic gaps of 0.2–3 s
  (`gap_aug_p`), matching the horizontal-component gaps ayzek measures on
  real data. A gap is never placed over the labelled onset.

## 4. Training

- **Dense supervision.** A loss on every 0.1 s token: 400 targets per 40 s crop
  instead of one per window. Causal masking trains every position in parallel.
- **No label smoothing** (§1).
- **Weights:**
  - The first second after P and the two seconds before it are weighted ×2.
    The first is the latency that matters; the second is where a premature
    trigger costs most.
  - Tokens within `p_tolerance_s` of P get weight 0. Their label is a coin flip
    at that label quality: 0.1 s for manual picks, 0.3 s for AIC-refined, 1.0 s
    for bare TauP. See DATA.md for how the 0.3 s was measured.
- **dt loss:** smooth-L1 on tokens after P, times `dt_weight` (0.1).
- **Crops.** A 40 s crop starts anywhere up to 1 s before P, so every crop holds
  P and at least a second of noise before it. Where P falls inside the crop
  otherwise doesn't matter: the model is causal.
- **Lead-in: a seamless splice.** 80% of the time, an event *or noise* trace
  gets 5–45 s of the same station's older noise joined in front:
  - The event trace first loses its first 1.5 s (the filter's start-up).
  - The lead-in is scaled per component to the RMS of the event's own pre-P
    noise.
  - The two are joined with a 0.5 s equal-power crossfade: no gap, no level
    step.
  - Noise traces get the same splice, so even a seam the model could find says
    nothing about the label.

  **Why the join must be seamless.** An FDSN event trace starts at origin
  time, so without a lead-in every onset comes 2–11 s after the data begins.
  The model learns that cue and then misses onsets on a continuous stream,
  where the data never begins.

  **Why a gap join failed.** The first version joined the lead-in across a
  0.2–1 s gap. That only moved the cue: onsets now came 2–11 s after the gap.
  On the same 8 events, that model fired 3–8 s after origin on the event trace
  alone, and 7–24 s after origin behind 24 h of continuous data, mostly on the
  S wave. Validation missed it because its traces also started at origin. On
  the splice-aware validation below, that model catches 4.2% of events within
  1 s instead of the 86% it reported.
- **A second event in the coda.** Every stored trace holds one event, so
  the model never saw an onset inside another event's coda and learned to
  read one as more coda: on the Marmara M6.2 sequence the v2 model's dt did
  not restart for 116 of 296 catalogued arrivals, and ayzek's trigger could
  not fire on them. `second_p` (0.5) of training event crops now get a second
  event trace added 6–30 s after the first P, from the same station where
  there is one, at 1–10× the RMS just before it (log-uniform), faded in over
  half a second. p stays 1 through it; dt restarts at its P and is not
  trained within its label tolerance; the geometry head is trained only
  before it, since its targets are the first event's (`data._second`).
  The restart itself is a few tokens against a coda of many, and the first
  models learned it only partly: on the Sındırgı M6.1 sequence dt fell about
  7 s at most missed aftershocks, but only to 2–3 s, and ayzek's restart
  needs 2 s. So the dt loss of the 3 s after the second P is weighted 5×
  (`second_dt_s`, `second_dt_weight`), and the second P now lands after dt
  has reached its 10 s cap in most crops, as in a sequence. Validation
  reports the median lowest dt within 3 s of each second P.
- **Sampling.** Half of each batch is noise traces and half event traces,
  whatever the group sizes. `--fallback` stores (STEAD) take
  `fallback_weight` of the event half.
- AdamW, cosine schedule with warm-up, bf16 autocast, gradient clip 1.0.

## 5. Evaluation

AUC is not reported. A stream detector is judged on two numbers that trade
off against each other through the threshold:

- **Latency on events**: recall within 0.25 / 0.5 / 1 / 2 / 4 s of P. The time
  of a token's output is the time of its last sample, the earliest moment a
  real system could know it. A crossing earlier than `P − tolerance` is an
  *early* trigger. It is reported, and it does not count as a detection.
- **False triggers per hour on noise**: triggers of ayzek's rule (below:
  rising edges with hysteresis, re-arming below half the threshold, plus dt
  restarts), over the hours actually monitored, excluding gaps.

**Validation splices too.** Every validation trace gets `eval_lead_in_s`
(40 s) of the station's older noise joined in front. That is longer than the
31.7 s lookback, so the model cannot see where the data begins, and the
numbers describe a continuous stream rather than a trace that starts at
origin. Traces whose station has no noise, or whose P is too early to splice,
stay as they are.

**The trigger is ayzek's.** Latency and false triggers are counted with the
trigger ayzek runs (`metrics.TriggerRule`, from `TrainConfig.dt_reset*` and
`min_trigger_gap_s`): a rising edge, or a restart of dt while p stays high,
and no trigger whose P date is within 15 s of the last one. Every 4th
validation event (`eval_second_every`) carries a second event in its coda at
a fixed draw, scored on its own ("second onsets"). The restart level is
1 s (`dt_reset_below`): once second onsets were trained harder, dt fell to
about 0.8 s at a new onset, and at 2 s the restart also fired in codas.
The export writes the rule into the model file, so ayzek triggers with the
rule the checkpoint was selected with. With `--dt-reset 0
--eval-second-every 0 --min-trigger-gap-s 0`, validation is what it was
before.

Both are swept over the threshold. The **operating point** is the lowest
threshold that keeps noise within `fa_target_per_hour` (default 1 per station
per hour). **Model selection** takes the checkpoint with the best recall within
1 s at that operating point, over first and second onsets together, so a
checkpoint is chosen for catching aftershocks as well as isolated events.

`onset replay` is the test that matters most. It runs a trained model over a
continuous recording, with the station context refreshed online from quiet
blocks, and matches triggers against catalogued arrivals. The 15
`day_before_24h` pulls joined to their `window_post_60s` event windows are the
continuous test set. They include GAZ and KMRS through the 2023-02-06 Mw 7.8.

## 6. Deployment (ayzek)

`stream.StreamingDetector` is the reference algorithm for the C++ port:

1. Keep the newest `ceil(RF / 10) × 10 = 30` conditioned samples. The stem's
   receptive field is 23 samples.
2. Every 10 new samples, run the stem over them and keep the last output token.
3. For each layer:
   - append the token's key and value to a ring of 80;
   - attend with ALiBi bias `−slope × distance`;
   - cross-attend to the 8 context tokens, whose keys and values are computed
     once per context refresh;
   - apply the feed-forward.
4. Read out p and dt.

**Cost per token** (every 0.1 s), in multiply-accumulates:

| part | per token |
|---|---:|
| stem | ~58k |
| attention projections | ~16k per layer |
| attention over 80 keys | ~10k per layer |
| cross-attention | ~9k per layer |
| feed-forward | ~33k per layer |
| **total** | **~330k** |

That is about 3.3 M per second per station, well below the 3-seed CNN–BiLSTM
detector ayzek runs now (5.2 ms per window, every 0.5 s). The context encoder
runs once a minute.

**New kernels ayzek needs:**
- causal padding of `k − s`;
- the ALiBi bias;
- a KV ring buffer;
- cross-attention to a fixed memory;
- the per-step channel LayerNorm, which is ayzek's existing `LayerNorm`
  applied along a different axis.

Conv1d, GELU, LayerNorm and multi-head attention already exist.

**Location from the geometry head.** A model trained with `--geometry 1`
also says, per token after P, how far away the event is (Gaussian in log km)
and in which direction (von Mises back-azimuth). ayzek uses that in place of
its S-P picker: every station sends its latest estimate once a second for
20 s after its trigger, and the network stage maximises the product of the
stations' likelihoods together with their P times (`locate.py`, transcribed
as ayzek's `pipeline/locate.cpp`). That gives a location from the trigger on
instead of after a 60 s picker window, and from a single station when it has
a back-azimuth. See ayzek's `docs/impl/15-geometry-location.md`.

## 7. Known limitations and open questions

- **TauP runs early here.** On the KO pulls the accepted AIC picks land a
  median ~1 s after the iasp91 prediction, so the AIC search window leans
  late (−2 / +3 s). A regional velocity model would shrink the 11% of traces
  that fall back to bare TauP labels.
- **The AIC tolerance was measured on STEAD, not on KO.** The median error is
  0.02 s and 96.7% of picks are within 0.3 s. STEAD's SNR and instruments
  differ from KO's, so treat 0.3 s as an estimate for KO.
- **Training context is 3 h old; deployed context is minutes old.** This is a
  mismatch in the easier direction.
- **Replay applies a context refresh at block boundaries**, including to a
  block's lead-in tokens. Scale changes are exact. Only the cross-attention
  memory for about 32 s of recomputed history differs from true streaming.
- **STEAD** has no noise traces to splice from, so its events still carry the
  "data just began" cue. Keep `fallback_weight` low, or splice from KO noise. Its preprocessing history (whether its band-pass was
  zero-phase) is unverified. If it was, STEAD onsets carry slight pre-P energy
  that FDSN onsets don't. It is a fallback for that reason.
- **No human-reviewed noise.** Noise windows are the `noise_pre_6h` pulls minus
  any window a visible catalogued arrival touches. Uncatalogued micro-events
  remain in them as label noise.
- **The picker and the magnitude regressor** are the next stages. The dt head
  exists to hand the picker its window. With the geometry head, location no
  longer needs the picker (§6).
- **The geometry head's uncertainties are the location's weights.** They are
  only as good as their calibration (`cal_1sd` in the evaluate geometry
  table, 0.68 when honest); an overconfident station pulls the location
  (`tests/test_locate.py`).
