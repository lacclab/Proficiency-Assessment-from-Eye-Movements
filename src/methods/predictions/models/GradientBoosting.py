import os

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

from src.constants import TREE_PARAM_GRIDS
from src.methods.predictions.models.BaseModel import BaseModel


class LightGBM(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        # LightGBM's feature_fraction, under its sklearn-API alias; the library
        # default of 1.0, exposed so the inner search can tune it.
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        n_jobs: int = int(os.environ.get("TREE_N_JOBS", "1")),  # threads per fit; set TREE_N_JOBS=8 on an idle machine
        verbose: int = -1,
    ):
        super().__init__(model_name="LightGBM")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

    # Keyed by the ModelNames *value*, not the enum: src.configs imports the
    # model classes, so importing ModelNames here would be circular.
    def hyperparameter_grid(self):
        return TREE_PARAM_GRIDS["LightGBM"]

    def fit(self, X, y):
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        # Wide per-text frames (~40k cols): pandas input costs a per-column
        # ingestion loop per fit (XGBoost also rebuilds feature names every
        # boosting round). Column order is fixed upstream (df[feature_cols]),
        # so hand the model one numpy block instead.
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)


class XGBoost(BaseModel):
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        n_jobs: int = int(os.environ.get("TREE_N_JOBS", "1")),  # threads per fit; set TREE_N_JOBS=8 on an idle machine
        verbosity: int = 0,
    ):
        super().__init__(model_name="XGBoost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=self.verbosity,
        )

    # Keyed by the ModelNames *value*, not the enum: src.configs imports the
    # model classes, so importing ModelNames here would be circular.
    def hyperparameter_grid(self):
        return TREE_PARAM_GRIDS["XGBoost"]

    def fit(self, X, y):
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        # Wide per-text frames (~40k cols): pandas input costs a per-column
        # ingestion loop per fit (XGBoost also rebuilds feature names every
        # boosting round). Column order is fixed upstream (df[feature_cols]),
        # so hand the model one numpy block instead.
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)
