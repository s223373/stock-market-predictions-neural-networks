import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from dataset_builder import BUY, HOLD, SELL, FORWARD_BARS

STATE_PATH = "nb_state.joblib"
CLASSES = np.array([SELL, HOLD, BUY])          # partial_fit needs this on the first call


def load_state():
    if os.path.exists(STATE_PATH):
        return joblib.load(STATE_PATH)
    return {"models": {}, "trained_through": None, "alpha": 1.0}


def save_state(state):
    joblib.dump(state, STATE_PATH)


def update_models(datasets, bool_cols, state, base_index, alpha=None):
    """Feed each per-offset model ONLY the blocks it hasn't seen yet."""
    bar = base_index.to_series().diff().median()
    new_end = base_index[-1] - FORWARD_BARS * bar     # labels need FORWARD_BARS to resolve
    last = state["trained_through"]
    alpha = alpha if alpha is not None else state["alpha"]

    for k, ds in datasets.items():
        m = np.asarray(ds.index < new_end)
        if last is not None:
            m &= np.asarray(ds.index >= last)
        if m.sum() == 0:
            continue
        if k not in state["models"]:
            state["models"][k] = BernoulliNB(alpha=alpha, binarize=None)
        state["models"][k].partial_fit(ds.loc[m, bool_cols], ds.loc[m, "target"],
                                       classes=CLASSES)

    state["trained_through"] = new_end
    state["alpha"] = alpha
    return state


if __name__ == "__main__":
    # build bunched datasets, update models, save state
    from bunched_datasets import build_bunched_datasets, META_COLS

    datasets, base = build_bunched_datasets(return_base=True)
    bool_cols = [c for c in datasets[0].columns if c not in META_COLS]
    state = load_state()
    state = update_models(datasets, bool_cols, state, base.index)
    save_state(state)
    models = state["models"]   # drop-in for score_blocks(models, ...)
# datasets, base = build_bunched_datasets(return_base=True)
# bool_cols = [c for c in datasets[0].columns if c not in META_COLS]
# state = load_state()
# state = update_models(datasets, bool_cols, state, base.index)
# save_state(state)
# models = state["models"]   # drop-in for score_blocks(models, ...)