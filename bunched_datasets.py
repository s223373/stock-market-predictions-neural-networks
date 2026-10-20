"""
bunched_datasets.py
===================
Builds "bunched" datasets: every sample = WINDOW consecutive 1m bars (default 5).
Signals fire on the 1m bars but are rolled up into the 5-bar sample.

Because the 5-bar grid can start at any of the 5 candles, there are WINDOW
datasets in total (offset 0..4), each one shifted by one 1m candle.

Roll-up rules (per 5-bar block)
-------------------------------
  boolean feature cols : 1 if it fired on ANY of the 5 bars (max)
  Close                : close of the LAST bar in the block
  target               : any BUY and no SELL   -> BUY  (2)
                         any SELL and no BUY   -> SELL (0)
                         BUY + SELL both       -> HOLD (1)
                         neither               -> HOLD (1)
  index                : timestamp of the LAST bar in the block

Usage
-----
  python bunched_datasets.py
"""

import numpy as np
import pandas as pd

from dataset_builder import (
    TICKER, PERIOD, INTERVAL,
    BUY, HOLD, SELL, LABEL_NAMES,
    _build_enriched, _clean_features, _is_binary_col, label_signals,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WINDOW            = 5       # 1m bars per sample (also = number of offset datasets)
DROP_GAPPED       = True    # drop blocks that span a data gap (overnight, missing bars)
OUTPUT_TEMPLATE   = "bunched_dataset_offset{k}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CORE
# ─────────────────────────────────────────────────────────────────────────────

def _bunch(df: pd.DataFrame, bool_cols: list[str], offset: int,
           window: int = WINDOW, bar: pd.Timedelta | None = None) -> pd.DataFrame:
    """Group df (bars with 'Close', bool_cols, 'target') into blocks of
    `window` bars, with the grid starting at row `offset`.
    `bar` = typical bar length, used only for the gap check."""
    sub = df.iloc[offset:]
    n_blocks = len(sub) // window
    if n_blocks == 0:
        print(f"  offset {offset}: only {len(sub)} bars available, need >= {window}")
        return pd.DataFrame()
    sub = sub.iloc[: n_blocks * window]          # drop incomplete trailing block

    idx = sub.index
    starts, ends = idx[0::window], idx[window - 1::window]   # keeps timezone

    # (n_blocks, window) views of the 1m data
    tgt = sub["target"].to_numpy(dtype=float).reshape(n_blocks, window)
    has_buy  = (tgt == BUY).any(axis=1)
    has_sell = (tgt == SELL).any(axis=1)
    has_nan  = np.isnan(tgt).any(axis=1)

    target = np.full(n_blocks, HOLD, dtype=float)
    target[has_buy & ~has_sell] = BUY
    target[has_sell & ~has_buy] = SELL

    feats = (sub[bool_cols].to_numpy(dtype=float)
             .reshape(n_blocks, window, len(bool_cols)).max(axis=1))

    out = pd.DataFrame(feats.astype(int), columns=bool_cols, index=ends)
    out.index.name = idx.name
    out.insert(0, "Close", sub["Close"].to_numpy().reshape(n_blocks, window)[:, -1])
    out["target"] = target
    out["start_ts"] = starts

    keep = ~has_nan
    n_nan = int(has_nan.sum())
    n_gap = 0
    if DROP_GAPPED and bar is not None:
        span_ok = np.asarray((ends - starts) <= window * bar)
        n_gap = int((keep & ~span_ok).sum())
        keep = keep & span_ok

    print(f"  offset {offset}: {n_blocks:,} blocks | dropped {n_nan:,} (NaN target) "
          f"+ {n_gap:,} (gaps) -> {int(keep.sum()):,} kept")

    out = out.loc[keep]
    out["target"] = out["target"].astype(int)
    return out


def build_bunched_datasets(
    ticker:   str = TICKER,
    period:   str = PERIOD,
    interval: str = INTERVAL,
    window:   int = WINDOW,
    return_base: bool = False,
):
    """Return {offset: bunched_dataframe} for offset in 0..window-1.
    If return_base=True, returns (datasets, base_1m_frame) instead."""
    print(f"\n{'═' * 55}\n  Building BUNCHED datasets ({ticker}, window={window})\n{'═' * 55}")

    # 1m features + 1m labels, computed ONCE and reused for all offsets
    enriched, feat_cols = _build_enriched(ticker, period, interval)
    enriched["target"] = label_signals(enriched)

    bool_cols = [c for c in feat_cols
                 if c != "Close" and c in enriched.columns
                 and _is_binary_col(enriched[c])]

    base = enriched[["Close"] + bool_cols + ["target"]].copy()
    base = _clean_features(base, bool_cols)      # NaN handling; target left alone

    if not bool_cols:
        raise ValueError("No boolean feature columns found - check which entries "
                         "are uncommented in feature_engineering.FEATURES.")

    bar = base.index.to_series().diff().median()
    print(f"\n  Base frame    : {len(base):,} bars  "
          f"({base.index.min()} -> {base.index.max()})")
    print(f"  Bar length    : {bar}   (expected {interval})")
    print(f"  Bool features : {len(bool_cols)}")
    print(f"  NaN targets   : {int(base['target'].isna().sum()):,}")
    if len(base) < window:
        print("  ⚠  Fewer bars than WINDOW - check PERIOD/INTERVAL/TICKER "
              "(yfinance limits 1m history to ~30 days, 7 days per request).")

    results = {}
    for k in range(window):
        ds = _bunch(base, bool_cols, offset=k, window=window, bar=bar)
        results[k] = ds
        _summarise(ds, k)
    return (results, base) if return_base else results


def _summarise(ds: pd.DataFrame, k: int) -> None:
    total = len(ds)
    print(f"\n  Offset {k}: {total:,} samples")
    if total == 0:
        return
    for code in (BUY, HOLD, SELL):
        cnt = (ds["target"] == code).sum()
        print(f"    {LABEL_NAMES[code]:4s} {cnt:7,}  ({100 * cnt / total:5.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = build_bunched_datasets()
    for k, ds in datasets.items():
        path = OUTPUT_TEMPLATE.format(k=k)
        ds.to_csv(path)
        print(f"✓  Saved offset {k} → {path}")