# plot_paths.py
import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot NEG/NEU/POS paths (and optional comparators)")
    p.add_argument("--csv", default="out/paths.csv", help="CSV exported by CLI --csv")
    p.add_argument("--json", default="out/live.json", help="JSON result for labels (optional)")
    p.add_argument("--show-indep", action="store_true", help="Plot non-reflexive comparator paths if present")
    p.add_argument("--save", default=None, help="Save figure to path instead of showing")
    return p.parse_args()


def get_first(row, keys):
    for k in keys:
        if k in row:
            return row[k]
    raise KeyError(keys[0])


def main():
    args = parse_args()

    neg, neu, pos = [], [], []
    neg_ind, pos_ind = [], []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            neg.append(float(get_first(row, ["NEG", "negative"])) if get_first(row, ["NEG", "negative"]) not in (None, "") else np.nan)
            neu.append(float(get_first(row, ["NEU", "neutralized"])) if get_first(row, ["NEU", "neutralized"]) not in (None, "") else np.nan)
            pos.append(float(get_first(row, ["POS", "positive"])) if get_first(row, ["POS", "positive"]) not in (None, "") else np.nan)
            # Optional comparators
            v_neg_i = row.get("NEG_INDEP") or row.get("neg_independent")
            v_pos_i = row.get("POS_INDEP") or row.get("pos_independent")
            if v_neg_i not in (None, ""):
                neg_ind.append(float(v_neg_i))
            if v_pos_i not in (None, ""):
                pos_ind.append(float(v_pos_i))

    x = np.arange(1, len(neu) + 1)

    # Band from POS-NEG spread (same heuristic as orchestrator)
    spread = np.abs(np.array(pos) - np.array(neg))
    band_w = float(np.nanmean(spread) / 4.0)
    upper = [round(v + band_w, 2) for v in neu]
    lower = [round(v - band_w, 2) for v in neu]

    unit = "days"
    try:
        if os.path.exists(args.json):
            with open(args.json, encoding="utf-8") as jf:
                meta = json.load(jf).get("meta", {})
                unit = meta.get("unit", unit)
    except Exception:
        pass

    plt.figure(figsize=(8, 4.8))
    plt.plot(x, neg, label="NEG", color="#d62728", lw=2)
    plt.plot(x, neu, label="NEU", color="#1f77b4", lw=2)
    plt.plot(x, pos, label="POS", color="#2ca02c", lw=2)
    plt.fill_between(x, lower, upper, color="#1f77b4", alpha=0.15, label="NEU band")

    if args.show_indep and neg_ind and pos_ind:
        xi = np.arange(1, max(len(neg_ind), len(pos_ind)) + 1)
        plt.plot(xi, neg_ind, label="NEG (indep)", color="#d62728", lw=1.5, ls=":")
        plt.plot(xi, pos_ind, label="POS (indep)", color="#2ca02c", lw=1.5, ls=":")

    plt.axhline(0, color="#888", lw=1, ls="--", alpha=0.6)
    plt.xlabel(f"Steps ({unit})")
    plt.ylabel("Pct move (%)")
    plt.title("NEG / NEU / POS paths with neutral variance band")
    plt.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()