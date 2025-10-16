#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_epoch_means(csv_path: Path, use_running_avg=False) -> pd.Series:
    """
    Return a Series indexed by epoch with the *per-epoch* training loss for one run.
    By default uses the raw per-batch loss ('loss_batch'); set use_running_avg=True to
    use 'loss_running_avg' instead.
    """
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        raise ValueError(f"{csv_path} missing 'epoch' column. Did you pass the batch_losses.csv produced earlier?")
    col = "loss_running_avg" if use_running_avg else "loss_batch"
    if col not in df.columns:
        raise ValueError(f"{csv_path} missing '{col}' column.")
    # Aggregate batches -> one value per epoch (simple mean; change to .median() if preferred)
    per_epoch = df.groupby("epoch")[col].mean().sort_index()
    per_epoch.name = csv_path.stem
    return per_epoch

def main():
    ap = argparse.ArgumentParser(description="Plot training loss history across runs with mean ± std per epoch.")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more batch_losses.csv files (one per run).")
    ap.add_argument("--use-running-avg", action="store_true",
                    help="Aggregate 'loss_running_avg' instead of 'loss_batch'.")
    ap.add_argument("--out", default="train_loss_mean_std.png", help="Output PNG path.")
    ap.add_argument("--agg-csv", default="train_loss_epoch_agg.csv",
                    help="Also write the aggregated table (per-epoch mean & std) here.")
    ap.add_argument("--show-per-run", action="store_true",
                    help="Overlay thin lines for each run.")
    args = ap.parse_args()

    # Load per-run epoch series
    series_list = []
    for p in args.csv:
        s = load_epoch_means(Path(p), use_running_avg=args.use_running_avg)
        series_list.append(s)

    # Join on epoch index (outer join keeps all epochs; NaNs handled in mean/std)
    per_run_df = pd.concat(series_list, axis=1).sort_index()

    # Mean & std across runs per epoch (skip NaN)
    agg = pd.DataFrame({
        "epoch": per_run_df.index,
        "mean_loss": per_run_df.mean(axis=1, skipna=True),
        "std_loss": per_run_df.std(axis=1, ddof=1, skipna=True),
        "n_runs": per_run_df.count(axis=1)
    })

    # Save aggregated CSV
    agg.to_csv(args.agg_csv, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    x = agg["epoch"].to_numpy()
    y = agg["mean_loss"].to_numpy()
    s = agg["std_loss"].to_numpy()

    # Mean line
    ax.plot(x, y, linewidth=2, label="Mean train loss")

    # ±1 std band
    ax.fill_between(x, y - s, y + s, alpha=0.2, label="±1 std")

    # Optional: overlay each run
    if args.show_per_run:
        for col in per_run_df.columns:
            ax.plot(per_run_df.index, per_run_df[col], linewidth=1, alpha=0.6, label=f"{col}")

    ax.set_xlabel("Epoch")
    ylabel = "Train loss (epoch mean of loss_running_avg)" if args.use_running_avg else "Train loss (epoch mean of loss_batch)"
    ax.set_ylabel(ylabel)
    ax.set_title("Training loss history: mean ± std across runs")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Avoid a huge legend if many runs
    if args.show_per_run:
        ax.legend(ncol=2, fontsize=8, frameon=False)
    else:
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Wrote figure: {args.out}")
    print(f"Wrote aggregated CSV: {args.agg_csv}")
    print("Per-epoch table (head):")
    print(agg.head().to_string(index=False))

if __name__ == "__main__":
    main()