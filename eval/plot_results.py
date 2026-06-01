"""Render the aggregate RAGAS scores from results.md into a bar chart.

    python -m eval.plot_results

Reads the committed numbers (no re-run needed) so the chart always matches the
table in results.md.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT

RESULTS = ROOT / "eval" / "results.md"
OUT = ROOT / "results" / "ragas_scores.png"

ORDER = ["Faithfulness", "Answer relevancy", "Context precision", "Context recall"]


def parse_scores(md):
    scores = {}
    for line in md.splitlines():
        m = re.match(r"^\|\s*([A-Za-z][A-Za-z ]*?)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", line)
        if m:
            scores[m.group(1).strip()] = float(m.group(2))
    return scores


def main():
    scores = parse_scores(RESULTS.read_text())
    labels = [m for m in ORDER if m in scores]
    values = [scores[m] for m in labels]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.barh(labels[::-1], values[::-1], color="#6E56CF", height=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title("RAGAS evaluation — CS2 RAG assistant")
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}  {dict(zip(labels, values))}")


if __name__ == "__main__":
    main()
