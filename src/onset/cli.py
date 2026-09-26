"""`onset <command> ...`: one entry point, each command with its own --help."""
from __future__ import annotations

import sys

COMMANDS = {
    "build-fdsn": ("onset.build_fdsn", "main", "FDSN catalogue pulls -> dataset store"),
    "build-stead": ("onset.build_stead", "build", "STEAD chunks -> dataset store (fallback)"),
    "validate-aic": ("onset.build_stead", "validate_aic",
                     "measure the AIC P refinement against STEAD manual picks"),
    "train": ("onset.train", "main", "train the detector"),
    "evaluate": ("onset.evaluate", "main", "latency and false triggers on a split"),
    "replay": ("onset.replay", "main", "run a trained detector over continuous miniSEED"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print("usage: onset <command> [args]\n")
        for name, (_, _, help_) in COMMANDS.items():
            print(f"  {name:<14s} {help_}")
        sys.exit(0 if not argv or argv[0] in ("-h", "--help") else 2)
    module, fn, _ = COMMANDS[argv[0]]
    getattr(__import__(module, fromlist=[fn]), fn)(argv[1:])
