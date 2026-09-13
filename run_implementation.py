#!/usr/bin/env python3
"""Run a retained EXP1/EXP2 implementation file with its local dependencies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXP1 = ROOT / "src/exp1_boundary_interpretability"
EXP1_ALGORITHMS = EXP1 / "algorithms"
EXP2 = ROOT / "src/exp2_reconstruction_utility"
PROFILES = {
    "exp1": [EXP1, EXP1_ALGORITHMS],
    "decoder": [EXP2 / "decoder_finetuning", EXP1_ALGORITHMS],
    "reconstruction": [EXP2 / "reconstruction", EXP2 / "decoder_finetuning", EXP1_ALGORITHMS],
    "direct-asr": [EXP2 / "direct_q1_asr", EXP2 / "reconstruction", EXP2 / "decoder_finetuning", EXP1_ALGORITHMS],
    "matched-rate": [EXP2 / "matched_rate", EXP2 / "reconstruction", EXP1_ALGORITHMS],
    "cross-rate": [EXP2 / "cross_rate", EXP2 / "reconstruction", EXP1_ALGORITHMS],
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=PROFILES, required=True)
parser.add_argument("script", type=Path)
parser.add_argument("args", nargs=argparse.REMAINDER)
args = parser.parse_args()
script = (ROOT / args.script).resolve()
if not script.is_relative_to(ROOT / "src") or not script.is_file():
    parser.error("script must be a retained Python file under src/")
environment = os.environ.copy()
environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in PROFILES[args.profile])
raise SystemExit(subprocess.call([sys.executable, str(script), *args.args], cwd=ROOT, env=environment))
