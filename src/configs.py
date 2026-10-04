"""The model registry: ModelNames and one ready-to-use instance per model (MODELS)."""
from enum import StrEnum

from src.methods.predictions.models.RidgeRegression import RidgeRegression, LogRidgeRegression
from src.methods.predictions.models.LinearRegression import LinearRegression
from src.methods.predictions.models.RandomForest import RandomForest
from src.methods.predictions.models.DecisionTree import DecisionTree
from src.methods.predictions.models.GradientBoosting import LightGBM, XGBoost
from src.methods.predictions.models.TabStar import TabStar
from src.methods.predictions.models.TabPFN import TabPFN
from src.methods.predictions.models.AvgModel import AvgModel


class ModelNames(StrEnum):
    RIDGE_CLASSIFIER = "Ridge_Classifier"
    LOG_RIDGE_REGRESSION = "Log_Ridge_Regression"
    LINEAR_REGRESSION = "Linear_Regression"
    RANDOM_FOREST = "Random_Forest"
    DECISION_TREE = "Decision_Tree"
    LIGHTGBM = "LightGBM"
    XGBOOST = "XGBoost"
    TABSTAR = "TabStar"
    TABPFN = "TabPFN"
    # Featureless floor: predicts the training fold's mean target. Runs
    # through the same folds as everything else; see AvgModel.
    AVERAGE = "Average"


MODELS = {
    ModelNames.RIDGE_CLASSIFIER: RidgeRegression(),
    ModelNames.LOG_RIDGE_REGRESSION: LogRidgeRegression(),
    ModelNames.LINEAR_REGRESSION: LinearRegression(),
    ModelNames.RANDOM_FOREST: RandomForest(),
    ModelNames.DECISION_TREE: DecisionTree(),
    ModelNames.LIGHTGBM: LightGBM(),
    ModelNames.XGBOOST: XGBoost(),
    # TabStar() is cheap to instantiate (model built lazily at fit time), so it
    # is safe here even when the optional `tabstar` package isn't installed.
    ModelNames.TABSTAR: TabStar(),
    # Same lazy-build pattern as TabStar — safe without the `tabpfn` package.
    ModelNames.TABPFN: TabPFN(),
    ModelNames.AVERAGE: AvgModel(),
}
