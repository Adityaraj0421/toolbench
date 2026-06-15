"""Regenerate the under-calling chart from raw traces.

Reads runs/under-calling/ and runs/under-calling-xmodels/ (produced by running
the two configs of the same name) and plots calculator-use rate per model for the
neutral vs discouraged tool description. Writes docs/under-calling.png.

Usage:
    pip install -e ".[viz]"
    python -m toolbench run experiments/under-calling.yaml          # needs OPENROUTER_API_KEY
    python -m toolbench run experiments/under-calling-xmodels.yaml
    python scripts/plot_under_calling.py
"""

import glob
import json
from collections import defaultdict

import matplotlib.pyplot as plt

RUN_DIRS = ["runs/under-calling", "runs/under-calling-xmodels"]
# ordered to tell the story left-to-right: robust/over-corrects -> collapses
ORDER = [
    "claude-3.5-haiku",
    "gpt-4o-mini",
    "deepseek-chat-v3-0324",
    "llama-3.3-70b-instruct",
    "mistral-small-3.2-24b-instruct",
    "gemini-2.5-flash",
]


def call_rates():
    agg = defaultdict(lambda: {"n": 0, "called": 0})
    for d in RUN_DIRS:
        for f in glob.glob(f"{d}/*.jsonl"):
            recs = [json.loads(line) for line in open(f)]
            meta = recs[0]
            called = any(r["type"] == "turn" and r["tool_calls"] for r in recs)
            model = meta["model"].split("/")[-1]
            a = agg[(model, meta["variant"])]
            a["n"] += 1
            a["called"] += called
    return {k: 100 * v["called"] / v["n"] for k, v in agg.items()}


def main():
    rates = call_rates()
    models = [m for m in ORDER if (m, "neutral") in rates]
    neutral = [rates[(m, "neutral")] for m in models]
    discouraged = [rates[(m, "discouraged")] for m in models]

    x = range(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar([i - w / 2 for i in x], neutral, w, label="neutral description", color="#4C78A8")
    b2 = ax.bar([i + w / 2 for i in x], discouraged, w,
                label='"prefer your own knowledge"', color="#E45756")

    ax.set_ylabel("calculator-use rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("One sentence in a tool description changes whether models use it\n"
                 "(hard multiplication all 6 models fail unaided; n=30 per bar)", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig("docs/under-calling.png", dpi=150, bbox_inches="tight")
    print("wrote docs/under-calling.png")


if __name__ == "__main__":
    main()
