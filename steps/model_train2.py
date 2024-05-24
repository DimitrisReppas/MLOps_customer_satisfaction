import sys
import os
# Append the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import logging

import mlflow
import pandas as pd
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

#from .config import ModelNameConfig
# # Import ModelNameConfig from config module directly
from config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker
experiment_tracker = "mlflow_tracker"



@step(experiment_tracker=experiment_tracker)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None

        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e
if __name__ == "__main__":
    from steps.clean_data import clean_data # Assuming clean_data is stored in a file called steps/clean_data.py
    from steps.ingest_data import ingest_data # Assuming ingest_data is stored in a file called steps/ingest_data.py
    data = ingest_data()
    # Call clean_data to get data subsets
    x_train, x_test, y_train, y_test = clean_data(data)  # Assuming 'data' is your original DataFrame
    
    # Get model config
    config = ModelNameConfig()
    
    # Call train_model with data subsets and config
    trained_model = train_model(x_train, x_test, y_train, y_test, config)