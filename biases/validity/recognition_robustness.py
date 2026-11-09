#!/usr/bin/env python3
"""Table 8: recognition robustness. Prominent (confluence) vs obscure (ofbiz) pick rate and their gap
on the recognition menu (L2_cve), across the 3 prompt framings + neutral baseline and temperatures
0.0/0.7/1.0. Reads data/recognition-robust-framing.csv + recognition-robust-temp.csv.
Rate = attraction convention: picks / all present trials in the setting.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMINENT = "confluence-2023-22527"
OBSCURE = "ofbiz-2024-45195"
RUNG = "L2_cve"

ROWS = [
    ("Baseline (neutral prompt, T=0.7)", "recognition-robust-framing.csv",
     {"prompt_id": "blind", "info_level": RUNG, "temperature": "0.7"}),
    ("Prompt: terse operator", "recognition-robust-framing.csv",
     {"prompt_id": "terse_operator", "info_level": RUNG}),
    ("Prompt: analyst", "recognition-robust-framing.csv",
     {"prompt_id": "analyst", "info_level": RUNG}),
    ("Prompt: chain-of-thought", "recognition-robust-framing.csv",
     {"prompt_id": "chain_of_thought", "info_level": RUNG}),
    ("Temperature 0.0", "recognition-robust-temp.csv",
     {"prompt_id": "blind", "info_level": RUNG, "temperature": "0.0"}),
    ("Temperature 1.0", "recognition-robust-temp.csv",
     {"prompt_id": "blind", "info_level": RUNG, "temperature": "1.0"}),
]


def rate(csv_name, arm, filt):
    """Attraction rate: picks(arm) / all present trials matching filt (percent)."""
    tot = pk = 0
    for r in csv.DictReader(open(HERE / "data" / csv_name)):
        if any(str(r.get(k)) != str(v) for k, v in filt.items()):
            continue
        tot += 1
        if r["choice_matched"] == arm:
            pk += 1
    return 100 * pk / tot if tot else 0


if __name__ == "__main__":
    print("Table 8: recognition robustness (recognition menu, L2_cve)\n")
    print(f"{'Setting':36} {'Prominent':>10} {'Obscure':>9} {'Gap (pts)':>10}")
    for label, csv_name, filt in ROWS:
        p = rate(csv_name, PROMINENT, filt)
        o = rate(csv_name, OBSCURE, filt)
        print(f"{label:36} {p:9.1f}% {o:8.1f}% {p - o:+9.1f}")
