#!/usr/bin/env python3
"""Drive backends from config/bench/*.yaml. Implemented with experiment jobs later."""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    args = ap.parse_args()
    print(f"run_experiment stub {args.yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
