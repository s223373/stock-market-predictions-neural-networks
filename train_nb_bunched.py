"""
train_nb_bunched.py
===================
Trains one BernoulliNB per bunched offset dataset and bundles everything the
live trader needs into a single checkpoint (models, bool_cols, window,
exit settings) - same idea as the LSTM checkpoint.

Usage
-----
    python train_nb_bunched.py
Output
------
    NBBunched_<timestamp>.joblib
"""

import time

import joblib
from sklearn.naive_bayes import BernoulliNB

from dataset_builder import TICKER, PERIOD, INTERVAL, FORWARD_BARS
from bunched_datasets import build_bunched_datasets, WINDOW
from nb_bunched_backtest import TARGET_PCT, STOP_PCT, MAX_HOLD_BARS

META_COLS = ["Close", "target", "start_ts"]


def main():
    datasets = build_bunched_datasets()

    bool_cols = [c for c in datasets[0].columns if c not in META_COLS]
    print(f"\n{len(bool_cols)} boolean features: {bool_cols}")

    models = {}
    for k, ds in datasets.items():
        model = BernoulliNB(binarize=None)           # features are already 0/1
        model.fit(ds[bool_cols], ds["target"])
        models[k] = model
        print(f"  offset {k}: trained on {len(ds):,} samples")

    checkpoint = {
        "models": models,                 # {offset: fitted BernoulliNB}
        "bool_cols": bool_cols,
        "window": WINDOW,
        "ticker": TICKER,
        "period": PERIOD,
        "interval": INTERVAL,
        "forward_bars": FORWARD_BARS,
        "target_pct": TARGET_PCT,
        "stop_pct": STOP_PCT,
        "max_hold_bars": MAX_HOLD_BARS,
    }

    path = f"NBBunched_{time.strftime('%Y-%m-%d_%H-%M-%S')}.joblib"
    joblib.dump(checkpoint, path)
    print(f"\n✓ Saved checkpoint → {path}")


if __name__ == "__main__":
    main()