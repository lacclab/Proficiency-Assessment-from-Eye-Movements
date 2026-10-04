import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.constants import TREE_PARAM_GRIDS
from src.methods.predictions.models.BaseModel import BaseModel


class DecisionTree(BaseModel):
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        super().__init__(model_name="DecisionTree")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # Single CART tree — no n_jobs knob (sklearn's DecisionTreeRegressor is
        # single-threaded), so TREE_N_JOBS doesn't apply here.
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )

    # Keyed by the ModelNames *value*, not the enum: src.configs imports the
    # model classes, so importing ModelNames here would be circular.
    def hyperparameter_grid(self):
        return TREE_PARAM_GRIDS["Decision_Tree"]

    def fit(self, X, y):
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        # Same numpy handoff as the other tree wrappers: avoids sklearn's
        # per-column DataFrame validation on ~40k-column per-text frames.
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)
