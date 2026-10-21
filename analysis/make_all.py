#!/usr/bin/env python3
"""Rebuild every paper table and figures 2 to 11 from results/.

No network and no cluster. Figure 1 is not produced: the rasters are not
in this repository. Run from the repository root:

    python analysis/make_all.py
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebuild import ROOT, evaluate, write_expected  # noqa: E402


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def write_tables(results):
    out = ROOT / "analysis" / "out" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in results:
        grouped[row["paper_item"]].append(row)
    combined = out / "all_tables.csv"
    with combined.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["paper_item", "quantity", "regenerated", "printed", "status", "note"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow({
                "paper_item": row["paper_item"],
                "quantity": row["quantity"],
                "regenerated": row["got"],
                "printed": row["expected"],
                "status": row["status"],
                "note": row["note"],
            })
    for item, rows in grouped.items():
        path = out / f"{_slug(item)}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["quantity", "regenerated", "printed", "status", "note"]
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "quantity": row["quantity"],
                    "regenerated": row["got"],
                    "printed": row["expected"],
                    "status": row["status"],
                    "note": row["note"],
                })
        tex = out / f"{_slug(item)}.tex"
        lines = [
            "% Regenerated from results/. A FAIL row keeps the printed string in the comment.",
            "\\begin{tabular}{ll}",
            "Quantity & Regenerated \\\\",
            "\\hline",
        ]
        for row in rows:
            quantity = row["quantity"].replace("_", "\\_")
            value = row["got"].replace("_", "\\_")
            lines.append(f"{quantity} & {value} \\\\")
        lines.append("\\end{tabular}")
        tex.write_text("\n".join(lines) + "\n")
    blocked = out / "figure1_blocked.txt"
    blocked.write_text(
        "Figure 1 is not regenerated. It needs the 90 m DEM, the land mask, "
        "the observed flood raster and results/e12 count rasters. Those files "
        "are not in this repository, and this script does not download them.\n"
    )
    return combined


def compile_diagrams():
    pdflatex = shutil.which("pdflatex")
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if pdflatex is None:
        print("pdflatex not found; diagram PDFs were not rebuilt")
        return
    diagrams = ROOT / "analysis" / "diagrams"
    for stem, dest in (("fig2", "fig2.pdf"), ("fig3", "fig3.pdf"), ("fig4", "fig4.pdf")):
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"{stem}.tex"],
            cwd=diagrams,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        shutil.copy(diagrams / f"{stem}.pdf", figures / dest)
        print(f"wrote figures/{dest}")


def plot_figures():
    subprocess.run(
        [sys.executable, str(ROOT / "analysis" / "plots.py"),
         "--csv", "results/bench.csv", "--figdir", "figures", "--resdir", "results"],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    write_expected()
    results = evaluate()
    write_tables(results)
    compile_diagrams()
    plot_figures()
    fails = [row for row in results if row["status"] != "PASS"]
    print(f"tables written, {len(results) - len(fails)} PASS, {len(fails)} FAIL")
    for row in fails:
        print(f"FAIL {row['paper_item']} {row['quantity']}: printed {row['expected']} got {row['got']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
