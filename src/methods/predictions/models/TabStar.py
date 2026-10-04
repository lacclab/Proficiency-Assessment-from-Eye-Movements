import numpy as np
import pandas as pd
from loguru import logger

from src.methods.predictions.models.BaseModel import BaseModel


class TabStar(BaseModel):
    """Wrapper around the TabSTAR tabular foundation model.

    TabSTAR (https://github.com/eyalgomel/tabstar) is a pretrained transformer
    for tabular prediction. Unlike the linear models here it does its OWN
    preprocessing (encoding, normalization) and consumes a DataFrame with
    meaningful column names, so we deliberately skip the z-scaling pipeline and
    inherit BaseModel's identity ``prepare_training_data`` / ``prepare_test_data``
    (which pass ``df[feature_cols]`` through as a named DataFrame).

    The ``tabstar`` package is an optional heavyweight dependency (torch +
    pretrained weights), so it is imported lazily inside ``_build_model`` — this
    keeps ``import src.configs`` (which instantiates every model) working even
    when tabstar isn't installed. The ImportError only fires if you actually
    train a TabStar model.
    """

    def __init__(
        self,
        device: str | None = None,
        verbose: bool = False,
        random_state: int = 42,
        **tabstar_kwargs,
    ):
        super().__init__(model_name="TabStar")
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        self.tabstar_kwargs = tabstar_kwargs
        self.model = None  # built lazily at fit time

    def _build_model(self):
        try:
            from tabstar import TabSTARRegressor
        except ImportError as e:
            raise ImportError(
                "TabStar requires the 'tabstar' package, which is not installed. "
                "Install it with `pip install tabstar` (pulls in torch + pretrained "
                "weights) to use this model."
            ) from e

        kwargs = dict(self.tabstar_kwargs)
        if self.device is not None:
            kwargs.setdefault("device", self.device)
        kwargs.setdefault("verbose", self.verbose)
        kwargs.setdefault("random_state", self.random_state)
        return TabSTARRegressor(**kwargs)

    def fit(self, X, y):
        # Keep X as a named DataFrame — TabSTAR uses column names as semantic
        # signal. y as a 1-D array/Series.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = y.squeeze() if isinstance(y, (pd.Series, pd.DataFrame)) else y
        self.model = self._build_model()
        logger.info(
            f"Fitting TabStar on {X.shape[0]} rows × {X.shape[1]} features"
        )
        self.model.fit(X, y)
        return self

    def _predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        preds = self.model.predict(X)
        return np.asarray(preds).ravel()
