#!/usr/bin/env python3
"""Table 9: scan-flag robustness. How often the flagged option (vsftpd-2011-2523-flagged) is picked on
the champions menu once the nmap scan is shown (L3_nmap), across the 3 prompt framings and temperatures
0.0/0.7/1.0. Reads data/h6-robustness.csv + h7-temperature.csv.
Rate = attraction convention: picks / all present trials in the setting.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLAGGED = "vsftpd-2011-2523-flagged"
RUNG = "L3_nmap"

ROWS = [
    ("Prompt: terse operator", "h6-robustness.csv",
     {"prompt_id": "terse_operator", "info_level": RUNG}),
    ("Prompt: analyst", "h6-robustness.csv",
     {"prompt_id": "analyst", "info_level": RUNG}),
    ("Prompt: chain-of-thought", "h6-robustness.csv",
     {"prompt_id": "chain_of_thought", "info_level": RUNG}),
    ("Temperature 0.0", "h7-temperature.csv",
     {"prompt_id": "blind", "info_level": RUNG, "temperature": "0.0"}),
    ("Temperature 0.7 (baseline)", "h7-temperature.csv",
     {"prompt_id": "blind", "info_level": RUNG, "temperature": "0.7"}),
    ("Temperature 1.0", "h7-temperature.csv",
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
    print("Table 9: scan-flag robustness (champions menu, L3_nmap)\n")
    print(f"{'Setting':32} {'Flagged picked':>14}")
    for label, csv_name, filt in ROWS:
        print(f"{label:32} {rate(csv_name, FLAGGED, filt):13.1f}%")
