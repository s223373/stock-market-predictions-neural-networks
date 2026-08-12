import numpy as np
import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class ForcedRootTree(BaseEstimator, ClassifierMixin):
    """
    Forces a chain of root splits on `force_cols` (each must be 0/1) before
    handing each resulting partition off to its own independently-fit
    DecisionTreeClassifier. Guarantees every column in force_cols is used
    at least once — regardless of whether it would have won on information
    gain — by making it structurally impossible to skip.

    len(force_cols) forced splits -> 2**len(force_cols) leaf branches, so
    keep force_cols short (2-3 combined boolean flags, not every raw
    column) or you'll starve each branch of training rows.

    Inherits BaseEstimator/ClassifierMixin so this drops into GridSearchCV
    exactly like DecisionTreeClassifier does — get_params/set_params come
    for free from BaseEstimator as long as __init__ only assigns args.
    """
    def __init__(self, force_cols=None, criterion="gini", max_depth=7,
                 min_samples_leaf=1, min_samples_split=2, max_features=None,
                 class_weight="balanced", random_state=42):
        self.force_cols = force_cols
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state

    def _tree_kwargs(self):
        return dict(
            criterion=self.criterion, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            class_weight=self.class_weight, random_state=self.random_state,
        )

    def _branch_key(self, X):
        """Tuple of 0/1 per force_col, one per row — identifies which of
        the 2**len(force_cols) leaf branches each row falls into."""
        return list(zip(*[X[c].astype(bool).astype(int).values for c in self.force_cols]))

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        keys = self._branch_key(X)
        self.branches_ = {}
        print(f"  ForcedRootTree branches (force_cols={self.force_cols}):")
        for combo in product([0, 1], repeat=len(self.force_cols)):
            mask = np.array([k == combo for k in keys])
            n_rows = int(mask.sum())
            n_classes = len(np.unique(y[mask])) if n_rows else 0
            label = ", ".join(f"{c}={v}" for c, v in zip(self.force_cols, combo))
            status = "OK" if (n_rows >= 2 and n_classes >= 2) else "FALLBACK (too little/no-variety data)"
            print(f"    [{label}] -> {n_rows:,} rows, {n_classes} classes  [{status}]")
            if n_rows < 2 or n_classes < 2:
                self.branches_[combo] = None
                continue
            self.branches_[combo] = DecisionTreeClassifier(**self._tree_kwargs()).fit(X[mask], y[mask])
        self.fallback_ = DecisionTreeClassifier(**self._tree_kwargs()).fit(X, y)
        return self

    def predict(self, X):
        keys = self._branch_key(X)
        out = np.empty(len(X), dtype=self.classes_.dtype)
        for combo in set(keys):
            idx = [i for i, k in enumerate(keys) if k == combo]
            tree = self.branches_.get(combo) or self.fallback_
            out[idx] = tree.predict(X.iloc[idx])
        return out

    @property
    def feature_importances_(self):
        """Averaged across all fitted branch trees, for plot_tree_feature_importances."""
        trees = [t for t in self.branches_.values() if t is not None]
        if not trees:
            return self.fallback_.feature_importances_
        return np.mean([t.feature_importances_ for t in trees], axis=0)

    @property
    def feature_names_in_(self):
        return self.fallback_.feature_names_in_