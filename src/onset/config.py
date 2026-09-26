"""Configuration for the model, the data pipeline and training.

Each config is a plain dataclass that round-trips through JSON, so a run
directory holds everything needed to rebuild what was trained: `config.json`
is read back by `evaluate` and `replay` rather than retyped as flags.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SAMPLE_RATE = 100.0


@dataclass
class ModelConfig:
    """Geometry of `OnsetDetector`.

    `stem` is a list of causal conv stages `(kernel, stride, channels)`. The
    product of the strides is the token period in samples: 10 at 100 Hz, so
    one token every 0.1 s.
    """

    in_channels: int = 4                 # Z, N, E, gap mask
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    window_tokens: int = 80              # 8 s of near past per attention layer
    stem: list = field(default_factory=lambda: [[7, 2, 32], [5, 1, 48], [5, 5, 64]])
    context_tokens: int = 8
    context_layers: int = 1
    dropout: float = 0.1
    max_dt_s: float = 10.0
    sample_rate: float = SAMPLE_RATE
    # 1 adds the geometry head: epicentral distance and back-azimuth, each
    # with its own uncertainty, for every token after P (model.py).
    geometry: int = 0

    @property
    def stride(self) -> int:
        """Samples per token."""
        return math.prod(s for _, s, _ in self.stem)

    @property
    def token_seconds(self) -> float:
        return self.stride / self.sample_rate

    @property
    def stem_receptive_field(self) -> int:
        """Input samples one stem output token depends on."""
        r = 1
        for k, s, _ in reversed(self.stem):
            r = (r - 1) * s + k
        return r

    @property
    def lookback_tokens(self) -> int:
        """Tokens one output depends on through the stacked sliding windows.

        Each layer reaches back `window_tokens - 1` further, so the model's
        effective lookback is `n_layers` times one window, not one window.
        """
        return self.n_layers * (self.window_tokens - 1) + 1


@dataclass
class DataConfig:
    """How a stored trace becomes one training or evaluation example."""

    seq_seconds: float = 40.0            # training crop; evaluation uses whole traces
    ctx_seconds: float = 60.0            # station context fed to the context encoder
    ctx_drop: float = 0.2                # train without context this often
    fallback_scale_s: float = 1.0        # scale from the crop's first second when no context
    gap_aug_p: float = 0.2               # probability of inserting a synthetic gap
    gap_aug_max_s: float = 3.0
    lead_in_p: float = 0.8               # splice older station noise in front (data.py)
    lead_in_max_s: float = 45.0
    eval_lead_in_s: float = 40.0         # evaluation: longer than the lookback
    splice_trim_s: float = 1.5           # drop the trace's filter start-up first
    splice_xfade_s: float = 0.5
    early_s: float = 1.0                 # first second after P is up-weighted ...
    early_weight: float = 2.0
    pre_s: float = 2.0                   # ... and so are the two seconds before it
    pre_weight: float = 2.0


@dataclass
class TrainConfig:
    epochs: int = 40
    steps_per_epoch: int = 1000
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    dt_weight: float = 0.1
    geo_weight: float = 0.1              # geometry head's share of the loss
    noise_fraction: float = 0.5          # of each batch, drawn from noise traces
    fallback_weight: float = 0.0         # share of event draws from --fallback sources
    fa_target_per_hour: float = 1.0      # operating point used for model selection
    num_workers: int = 6
    seed: int = 42


def _from_dict(cls, d: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


def save_run_config(run_dir: Path, model: ModelConfig, data: DataConfig,
                    train: TrainConfig, **extra) -> Path:
    p = Path(run_dir) / "config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model": asdict(model), "data": asdict(data),
                             "train": asdict(train), **extra}, indent=2))
    return p


def load_run_config(run_dir: Path) -> tuple[ModelConfig, DataConfig, TrainConfig, dict]:
    d = json.loads((Path(run_dir) / "config.json").read_text())
    return (_from_dict(ModelConfig, d["model"]), _from_dict(DataConfig, d["data"]),
            _from_dict(TrainConfig, d["train"]),
            {k: v for k, v in d.items() if k not in ("model", "data", "train")})
