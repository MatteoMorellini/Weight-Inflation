#!/usr/bin/env python3
import re, argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

PATTERN = re.compile(
    r"^\[(?P<ts>[^]]+)\].*?Epoch:\s*\[(?P<epoch_cur>\d+)\/(?P<epoch_tot>\d+)\]\t"
    r"Iter:\s*\[(?P<iter_cur>\d+)\/(?P<iter_tot>\d+)\]\t"
    r"Loss\s+(?P<loss_batch>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\((?P<loss_avg>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)"
)

def main():
    ap = argparse.ArgumentParser(description="Extract per-batch losses from a training log.")
    ap.add_argument("--log", required=True, help="Path to logger.log")
    #ap.add_argument("--out", default="batch_losses.csv", help="Output CSV path")
    args = ap.parse_args()
    out = os.path.join(os.path.dirname(args.log), 'batch_losses.csv')

    rows = []
    with Path(args.log).open("r", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m:
                continue
            g = m.groupdict()
            try:
                ts = datetime.strptime(g["ts"], "%Y-%m-%d %H:%M:%S,%f")
            except Exception:
                ts = g["ts"]  # keep raw if parsing fails
            rows.append({
                "timestamp": ts,
                "epoch": int(g["epoch_cur"]),
                "epoch_total": int(g["epoch_tot"]),
                "iter": int(g["iter_cur"]),
                "iter_total": int(g["iter_tot"]),
                "loss_batch": float(g["loss_batch"]),
                "loss_running_avg": float(g["loss_avg"]),
            })

    df = pd.DataFrame(rows).sort_values(["epoch", "iter"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()

# call script using: python extract_batch_losses.py --log /path/to/logger.log --out batch_losses.csv