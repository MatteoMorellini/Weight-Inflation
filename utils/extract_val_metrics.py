#!/usr/bin/env python3
import re, argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import os

#!/usr/bin/env python3
import re, argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

# Flexible pattern:
# - grabs timestamp
# - optional "(tag)" like "(brats-met)" or "(brats-met best)"
# - accepts "Epoch: 4" anywhere before val_loss
# - accepts "pixel auroc" OR "pixel_auroc"
VAL_LINE = re.compile(
    r"""^\[(?P<ts>[^\]]+)\].*?           # [timestamp]
        (?:\((?P<tag>[^)]+)\))?          # optional (tag)
        .*?Epoch:\s*(?P<epoch>\d+)       # Epoch: N (anywhere before metrics)
        .*?image\W*auroc:\s*(?P<img>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*
        pixel[_\s]*auroc:\s*(?P<pixel>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*
        val_loss:\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

def main():
    ap = argparse.ArgumentParser(description="Extract per-epoch validation metrics from a training log.")
    ap.add_argument("--log", required=True, help="Path to logger.log")
    args = ap.parse_args()
    out = os.path.join(os.path.dirname(args.log), 'val_metrics.csv')
    rows, matched, near = [], 0, 0
    p = Path(args.log)
    with p.open("r", errors="ignore") as f:
        for line in f:
            if "val_loss" not in line:
                continue  # speed: only lines that likely contain validation summary
            m = VAL_LINE.search(line)
            if not m:
                near += 1
                continue
            g = m.groupdict()
            # timestamp parse
            try:
                ts = datetime.strptime(g["ts"], "%Y-%m-%d %H:%M:%S,%f")
            except Exception:
                ts = g["ts"]
            rows.append({
                "timestamp": ts,
                "tag": (g["tag"] or "").strip(),
                "epoch": int(g["epoch"]),
                "image_auroc": float(g["img"]),
                "pixel_auroc": float(g["pixel"]),
                "val_loss": float(g["val"]),
            })
            matched += 1

    df = pd.DataFrame(rows)
    if not df.empty:
        # If multiple lines per epoch, keep the last chronologically
        df = (df.sort_values(["epoch", "timestamp"])
                .drop_duplicates(subset=["tag","epoch"], keep="last")
                .reset_index(drop=True))
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()