# onset: magnitude from peak P-wave displacement

This document describes how the magnitude of an event is estimated from the
peak displacement of the P wave (Pd) at the stations that record it, how the
relation between Pd, magnitude and distance was calibrated on the stored
event traces, and how the estimator performs. The measurement is
implemented in `src/onset/pd.py`, the calibration and evaluation in
`src/onset/pd_fit.py`, and the check on a continuous recording in
`src/onset/pd_replay.py`. ayzek evaluates the same estimator in real time
(its `docs/impl/17-pd-magnitude.md`).

---

## 1. Motivation

A regression from waveform to magnitude was attempted earlier
(`sismokaos/cascade_impl`). It reached a mean absolute error of about 0.4
magnitude units overall but underestimated the largest events, of which the
training data contain few: `fdsn_wide_x` holds 11,806 events of M 2 or more
but only 16 of M 5 or more. A learned regressor has no means of
extrapolating beyond the range it has seen, and resampling the few large
events adds no information.

The detector's input, moreover, carries no absolute amplitude. Each
component is divided by the station's noise RMS and band-passed at
1–45 Hz (`conditioning.py`, `dsp.py`). Both operations are appropriate for
detection: the first cancels the instrument gain and expresses the signal
relative to the station's own background, and the second removes the
microseism. Both also remove what magnitude depends on. Division by the
noise level removes the absolute scale, and the band-pass removes most of
the energy below 1 Hz, where the spectra of large earthquakes depart from
those of small ones.

The approach adopted here therefore separates the two tasks. Amplitude is
measured directly, on the raw vertical component with the instrument
response removed, in a separate processing chain. Magnitude follows from an
empirical scaling relation between Pd, magnitude and distance in which the
magnitude dependence has the form of a small number of slope parameters.
Every event trace contributes to the calibration of that relation, and its
extrapolation to large magnitudes rests on the slopes rather than on
examples of large events.

## 2. Measurement

For every event trace of a store, the raw counts of the vertical component
are converted to displacement by a causal chain (`pd.displacement`):

1. division by the channel's sensitivity at the time of the event, from its
   StationXML response (`ONSET_FDSN_ROOT/raw/data/station_inventory`), which
   yields ground velocity in m/s;
2. a two-pole Butterworth high-pass at 0.075 Hz;
3. integration to displacement;
4. the same high-pass again, which suppresses the drift that integration
   introduces.

Peak displacement is then

    Pd(τ) = max |d(t)|,   P ≤ t < P + τ,

for windows τ = 1, 2, 3, 4, 5, 7 and 10 s after the trace's labelled P.
The same quantity over the 10 s ending 1 s before P is the noise level
`pd_noise`. The chain starts 41 s before P, which is long enough for the
filters' transients to decay. All operations are causal, so Pd(τ) is
available at time P + τ in real time; ayzek computes it from the P time that
the transformer dates.

`onset measure-pd` applies this to every event trace of a store. On
`fdsn_wide_x`, 119,489 of 121,406 traces were measured; 860 were rejected
for a gap in the window and 1,057 because no response was available for the
channel.

## 3. Model

For each window τ the relation is

    log10 Pd_ij = α + β · m(M_j) + γ · r(R_ij) + s_i + ε_ij,   ε ~ N(0, σ²),

where i indexes stations and j events, and:

- m(M) = [M, max(0, M − 4)] is piecewise linear in magnitude, with a knot at
  M 4;
- r(R) = [log10 R, max(0, log10 R/70 km), max(0, log10 R/140 km)] is
  piecewise linear in the logarithm of hypocentral distance, with knots at
  70 and 140 km;
- R = (D² + h²)^½, with D the epicentral distance and a fixed depth h of
  10 km. The fixed depth matches deployment, where D is supplied by the
  geometry head and the depth is unknown;
- s_i is a station term, constrained to sum to zero, that absorbs site
  amplification and errors in the response metadata.

Each of these terms was introduced in response to a measured deficiency of
the simpler model, as described in section 5.

### 3.1 Censoring

Displacement emphasises the long periods at which the microseism is
strongest. The median pre-P noise level on `fdsn_wide_x` is 1.4 × 10⁻⁷ m,
which is comparable to the Pd of a small regional event. Of the values at
τ = 3 s, only 2–5% of those of events up to M 3 exceed three times the noise
level, against 75–90% above M 4.

A value is treated as measured only if it exceeds three times the noise
level (`MIN_SNR`). Otherwise it is treated as censored: it supplies the
upper bound log10(3 · pd_noise) on log10 Pd. The relation is fitted by
maximum likelihood with censoring (a Tobit model). A measured value
contributes the normal density, and a censored value contributes
Φ((c − μ)/σ), the probability that the value lies below its bound c. Each
event carries a total weight of one, shared equally among its stations, so
that the many stations of a well-recorded event do not dominate the fit.
The fit uses the events of M 2 or more in the train split.

### 3.2 Event magnitude

The magnitude of an event is estimated from all of its stations, measured
and censored alike. The latter bound the estimate from above, so that a
small event that is recorded above the noise at a single station is not
assigned that station's magnitude. The likelihood is combined with a
Gutenberg–Richter prior, p(M) ∝ 10^(−bM), and the estimate is the posterior
mean, with the posterior standard deviation as its stated uncertainty. Both
are evaluated on a grid of M from 1.0 to 8.5 in steps of 0.01. The b-value,
0.84, is the maximum-likelihood estimate (Aki 1965) from the train events
above the completeness magnitude of M 2.5; it varies by no more than 0.01
for completeness magnitudes between 2.3 and 2.7. An event that no station
records above the threshold receives no estimate.

The prior corrects a selection effect. A small event receives an estimate
only when some station records it above the noise, and the small events
that satisfy this condition are disproportionately those whose amplitudes
came out high. Their maximum-likelihood estimates are therefore biased
upward. Since small events greatly outnumber large ones, an amplitude near
the noise level is more probably a small event recorded high than a larger
event recorded low, and the prior expresses exactly this. For large events
the likelihood is sharply peaked and the prior has little effect.

## 4. Fitted relation

| τ | α | β (M) | β (M > 4) | γ (log R) | γ (> 70 km) | γ (> 140 km) | σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 s | −5.894 | 0.912 | −0.015 | −2.062 | +0.974 | −3.041 | 0.37 |
| 3 s | −5.579 | 0.925 | +0.123 | −2.188 | +1.557 | −3.072 | 0.32 |
| 5 s | −5.125 | 0.907 | +0.231 | −2.346 | +1.767 | −2.351 | 0.30 |
| 10 s | −5.388 | 0.879 | +0.350 | −2.025 | +0.706 | −0.814 | 0.29 |

Pd is in metres and R in kilometres. Every window is fitted to 96,958
values from 11,806 events, of which 83,041 (τ = 10 s) to 90,546 (τ = 1 s)
are censored. The slope above M 4 increases with the window, from about
zero at τ = 1 s to 0.35 at τ = 10 s. This is consistent with the source
spectrum: the low-frequency energy of the larger events enters the
measurement only as the window lengthens. Station terms were estimated for
125 stations, with a standard deviation of 0.47. Four stations have terms
beyond ±1 (CTKS −2.18, DAT −3.31, TOKT −1.74, YEDI −1.60), which
correspond to factors of 40–2000 in amplitude. These are taken to be
errors in the response metadata or a defective vertical channel, and the
stations are excluded from the estimates.

## 5. Model development

The model was arrived at in four steps, each prompted by a failure of the
preceding one. The diagnostic at each step was the calibration of the
fitted model on its own training data, by magnitude and distance band: the
observed fraction of censored values against the fraction the model
predicts, and the mean residual of the measured values against the mean
the model predicts for them (the mean of a normal truncated at the
threshold). Under a correct model the two members of each pair agree.
`onset fit-pd` writes this comparison to `calibration.csv`.

1. **Least squares on the measured values.** This fit uses only about 10%
   of the values, and among the small events it retains only those recorded
   high. The fitted magnitude slope is consequently flattened (β ≈ 0.63),
   and the relation, extrapolated with that slope, overestimates events of
   M 5 and above by about one magnitude unit.
2. **Censored likelihood, linear in M and log R.** The slope steepens to
   β ≈ 0.95 and the overestimate above M 5 falls to about 0.4. The
   calibration shows two remaining deficiencies: an excess of measured
   amplitude at 100–150 km and a deficit at 30–60 km, and an excess above
   M 4.5. The magnitude excess does not arise from the catalogue's change
   of scale from ML to Mw near M 4: at equal magnitude, ML and Mw events
   have the same residuals (excess −0.015 against −0.032 for M 3.5–4, and
   +0.010 against +0.038 for M 4–4.5).
3. **Knots in distance at 70 and 140 km.** The distance bands then agree
   to within 0.04 in residual and 0.02 in censored fraction.
4. **A knot in magnitude at M 4.** The excess above M 4.5 falls from 0.20
   to 0.15, and the bias above M 5 falls to about 0.1. The
   maximum-likelihood event estimates nonetheless remain biased upward for
   small events (+0.3 to +0.6 below M 3), although the calibration of the
   model is good in that range. The bias is therefore a property of the
   estimate rather than of the model, and the Gutenberg–Richter prior of
   section 3.2 removes most of it.

## 6. Results

The tables give event estimates for the stations of the test split (1,581
to 2,666 events, depending on the window). `n` is the number of events
with an estimate at τ = 5 s. The bias is the mean of the estimated minus
the catalogue magnitude, and the MAE its mean absolute value.

**Censored likelihood with the Gutenberg–Richter prior (the estimator
adopted):**

| band | n | bias 1 s | bias 3 s | bias 5 s | bias 10 s | MAE 1 s | MAE 3 s | MAE 5 s | MAE 10 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M < 3 | 1209 | +0.33 | +0.20 | +0.11 | +0.06 | 0.40 | 0.30 | 0.26 | 0.26 |
| M 3–4 | 900 | +0.02 | +0.01 | −0.03 | −0.06 | 0.26 | 0.21 | 0.20 | 0.20 |
| M 4–5 | 154 | −0.11 | +0.04 | +0.05 | +0.02 | 0.33 | 0.22 | 0.21 | 0.19 |
| M ≥ 5 | 11 | −0.05 | +0.13 | +0.06 | −0.07 | 0.22 | 0.16 | 0.12 | 0.14 |
| all | 2274 | +0.16 | +0.11 | +0.05 | +0.01 | 0.33 | 0.26 | 0.24 | 0.23 |

**Bias at M ≥ 5 for all four estimators (test split):**

| estimator | 1 s | 3 s | 5 s | 10 s |
|---|---:|---:|---:|---:|
| censored likelihood, Gutenberg–Richter prior | −0.05 | +0.13 | +0.06 | −0.07 |
| censored likelihood, maximum likelihood | +0.15 | +0.24 | +0.14 | 0.00 |
| least squares on the measured values, inverted | +0.30 | +0.91 | +0.98 | +0.99 |
| regression of M on log Pd and log R | −0.66 | −0.32 | −0.24 | −0.32 |

The regression of M on log Pd minimises the error in M, and its slope is
therefore attenuated by the ratio of the variance of the magnitudes to the
total variance (regression dilution). The resulting underestimate of the
largest events is the failure observed in the earlier regressor. The
adopted estimator is nearly unbiased from M 3 upward at every window of
3 s or more.

On the validation split the adopted estimator gives similar results above
M 3 (bias between −0.06 and +0.02, MAE 0.13–0.27). Below M 3, however, a
bias of +0.24 to +0.33 remains, against +0.06 to +0.33 on the test split.
These events lie close to the detection threshold, where the estimate is
most sensitive to the distribution of station distances and noise levels,
and that distribution differs between the station-disjoint splits.

### 6.1 A large event: Pazarcık, 2023-02-06, Mw 7.7

`onset pd-replay` was applied to the continuous recording of event 543430,
which covers the 24 hours that precede 01:36:28 UTC on 2023-02-06 at the
stations GAZ and KMRS. GAZ has no data between 00:31 and 01:35. KMRS, at
29 km from the AFAD epicentre of the Mw 7.7 mainshock (01:17:32), gives the
following estimates:

| window | 2 s | 3 s | 4 s | 5 s | 7 s | 10 s |
|---|---|---|---|---|---|---|
| estimate | 4.0 ± 0.4 | 5.3 ± 0.3 | 5.2 ± 0.3 | 5.7 ± 0.3 | 5.8 ± 0.2 | 5.9 ± 0.2 |

No sample in these windows approaches the digitiser's full scale, so the
underestimate is not caused by clipping. It is the saturation of short-window
amplitude measures for great earthquakes. The rupture lasted on the order
of a minute, and the first 10 s of the P wave record only the part of it
that had occurred by then. No relation calibrated on Pd(τ ≤ 10 s) can
recover Mw 7.7 from such a measurement. A streaming estimate rises as the
window lengthens, but early estimates for events above about M 6.5 must be
read as lower bounds. The aftershocks of the following 19 minutes received
no estimate, because their pre-P "noise" is the coda of the mainshock and
all of their values are censored. This is the correct behaviour of the
estimator, but it means that magnitudes are unavailable for aftershocks
inside the coda of a larger event.

## 7. Limitations

- The fitted relation is valid for the magnitudes of the train split
  (M 2–6.1) and for windows of up to 10 s. Above M 6, and for every event
  whose rupture outlasts the window, the estimate is a lower bound (6.1).
- Only 11 events of M 5 or more are in the test split, and 8 in the
  validation split. The errors quoted for that band are correspondingly
  uncertain.
- The catalogue reports ML below about M 4 and Mw above. The estimates
  therefore follow the catalogue's mixed scale. No difference between the
  scales is resolved in the residuals (section 5).
- The fixed depth of 10 km is incorrect for deep events, and the
  distance term absorbs part of that error.
- Four stations are excluded on the evidence of their station terms. Their
  response metadata have not yet been checked.
- Station terms are applied only where at least 20 values were available
  in the train split. A station outside that set is assumed to have a term
  of zero.

## 8. Reproduction

```bash
uv run --extra build onset measure-pd --data datasets/fdsn_wide_x           # -> datasets/fdsn_wide_x/pd.csv
uv run --extra build onset fit-pd --pd datasets/fdsn_wide_x/pd.csv --out runs/pd_v4
R=$ONSET_FDSN_ROOT/raw/data/batched_waveforms
uv run --extra build onset pd-replay $R/day_before_24h/event_543430_raw.mseed \
    $R/window_post_60s/event_543430_raw.mseed --fit runs/pd_v4/fit.json
```

`runs/pd_v4/fit.json` holds the coefficients, station terms and b-value
for every window, in the form that ayzek's export reads. `eval.csv` holds
the evaluation of section 6, and `calibration.csv` the calibration of
section 5.
