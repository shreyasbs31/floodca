#!/usr/bin/env python3
"""Compare regenerated numbers with the strings printed in the paper.

Reads analysis/expected_values.csv and recomputes each value from results/.
Prints PASS or FAIL for every value. Does not edit results or the expected file.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild import ROOT, build, fmt_like  # noqa: E402


def main() -> int:
    expected_path = ROOT / "analysis" / "expected_values.csv"
    if not expected_path.is_file():
        sys.stderr.write("missing analysis/expected_values.csv; run analysis/make_all.py\n")
        return 1
    computed = {(row["paper_item"], row["quantity"]): row for row in build()}
    fails = 0
    total = 0
    with expected_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            key = (row["paper_item"], row["quantity"])
            source = computed.get(key)
            expected = row["expected"]
            if source is None or source["raw"] is None:
                got = "MISSING"
                status = "FAIL"
            else:
                got = fmt_like(source["raw"], expected)
                status = "PASS" if got == expected else "FAIL"
            note = row.get("note") or (source or {}).get("note") or ""
            line = f"{status}  {row['paper_item']}  {row['quantity']}  printed {expected}  got {got}"
            if note and status == "FAIL":
                line += f"  ({note})"
            print(line)
            if status == "FAIL":
                fails += 1
    print(f"\n{total - fails} PASS, {fails} FAIL, {total} values")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
