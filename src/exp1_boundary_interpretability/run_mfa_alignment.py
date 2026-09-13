#!/usr/bin/env python3
"""Run MFA 3.x word+phone alignment for prepared TIMIT corpus."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def run_cmd(cmd: List[str], allow_fail: bool = False) -> int:
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")
    return proc.returncode


def ensure_mfa() -> None:
    if shutil.which("mfa") is None:
        raise RuntimeError(
            "`mfa` not found. Install first: bash exps/exp1_gt_boundaries/scripts/install_mfa.sh"
        )


def download_models(dictionary_model: str, acoustic_model: str) -> None:
    run_cmd(["mfa", "model", "download", "dictionary", dictionary_model], allow_fail=True)
    run_cmd(["mfa", "model", "download", "acoustic", acoustic_model], allow_fail=True)


def try_align(
    corpus_dir: Path,
    output_dir: Path,
    dictionary_model: str,
    acoustic_model: str,
    jobs: int,
    clean: bool,
) -> bool:
    download_models(dictionary_model, acoustic_model)

    cmd = [
        "mfa",
        "align",
        str(corpus_dir),
        dictionary_model,
        acoustic_model,
        str(output_dir),
        "--output_format",
        "long_textgrid",
        "--num_jobs",
        str(jobs),
    ]
    if clean:
        cmd.append("--clean")

    return run_cmd(cmd, allow_fail=True) == 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dictionary-model", type=str, default="")
    parser.add_argument("--acoustic-model", type=str, default="")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--no-clean", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ensure_mfa()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[str, str]]
    if args.dictionary_model and args.acoustic_model:
        pairs = [(args.dictionary_model, args.acoustic_model)]
    else:
        # Common MFA English model names across 3.x releases.
        pairs = [
            ("english_us_arpa", "english_us_arpa"),
            ("english_mfa", "english_mfa"),
        ]

    for dictionary_model, acoustic_model in pairs:
        print(f"[run_mfa_alignment] Trying dictionary={dictionary_model}, acoustic={acoustic_model}")
        ok = try_align(
            corpus_dir=args.corpus_dir,
            output_dir=args.output_dir,
            dictionary_model=dictionary_model,
            acoustic_model=acoustic_model,
            jobs=args.jobs,
            clean=not args.no_clean,
        )
        if ok:
            print("[run_mfa_alignment] Alignment done.")
            return
        print("[run_mfa_alignment] Failed with this model pair, trying next.")

    raise RuntimeError(
        "Alignment failed for all model pairs. Provide explicit --dictionary-model and --acoustic-model."
    )


if __name__ == "__main__":
    main()
