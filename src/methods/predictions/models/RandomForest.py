import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.constants import TREE_PARAM_GRIDS
from src.methods.predictions.models.BaseModel import BaseModel


class RandomForest(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        # sklearn's default is 1.0 — every split scans every feature, which is
        # bagged trees, not a random forest.
        #
        # Measured on MECO (LOPO, aug_p=1.0), 'sqrt' vs 1.0:
        #   scalar S_CLUSTERS  (240 cols, n=1005): r +0.5832 vs +0.5775, 2.9x faster
        #   per-text TRANSITIONS (41k cols, n=120): r +0.4873 vs +0.5233, 27.6x faster
        # So it is a genuine trade, not a free win: on the wide per-text frames
        # 'sqrt' samples only ~203 of 41k columns and, with the data 97.6%
        # zeros, most of those carry no signal — costing ~0.036 r. Accepted
        # deliberately: 1.0 would take RF per-text from 77 to ~6,500 core-hours
        # (a ~6-day study becomes ~40), and RF is a minor share of the grid.
        # Pinned rather than tuned so no config can reintroduce that blowup.
        max_features="sqrt",
        random_state: int = 42,
        # Threads PER fit. Default 1 (safe on a busy machine: these run inside a
        # ThreadPoolExecutor with ~16 concurrent jobs × LOPO loops of hundreds of
        # refits, so n_jobs=-1 caused 2000+ thread thrash). Set env TREE_N_JOBS=N
        # to give each fit N threads when the machine is idle — e.g. TREE_N_JOBS=8
        # with --n-jobs ~5 uses ~40 cores with no thrash and ~8× faster fits.
        n_jobs: int = int(os.environ.get("TREE_N_JOBS", "1")),
    ):
        super().__init__(model_name="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    # Keyed by the ModelNames *value*, not the enum: src.configs imports the
    # model classes, so importing ModelNames here would be circular.
    def hyperparameter_grid(self):
        return TREE_PARAM_GRIDS["Random_Forest"]

    def fit(self, X, y):
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        # Same numpy handoff as the boosting wrappers: avoids sklearn's
        # per-column DataFrame validation on ~40k-column per-text frames.
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)
