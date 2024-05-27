import sys
import os
# Append the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import logging

import mlflow
import numpy as np
import pandas as pd
from model.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
#from zenml.steps import Output, step
from zenml.client import Client
from typing import Tuple

client = Client()
client.activate_stack('mlflow_stack')
# Retrieve the experiment tracker from the active stack
active_stack = client.active_stack
experiment_tracker = active_stack.experiment_tracker



@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        # prediction = model.predict(x_test)
        # evaluation = Evaluation()
        # r2_score = evaluation.r2_score(y_test, prediction)
        # mlflow.log_metric("r2_score", r2_score)
        # mse = evaluation.mean_squared_error(y_test, prediction)
        # mlflow.log_metric("mse", mse)
        # rmse = np.sqrt(mse)
        # mlflow.log_metric("rmse", rmse)

        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        import pdb; pdb.set_trace()
        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise e
    
if __name__ == "__main__":
    from steps.clean_data import clean_data # Assuming clean_data is stored in a file called steps/clean_data.py
    from steps.ingest_data import ingest_data # Assuming ingest_data is stored in a file called steps/ingest_data.py
    from steps.model_train import train_model
    from config import ModelNameConfig
    data = ingest_data()
    # Call clean_data to get data subsets
    x_train, x_test, y_train, y_test = clean_data(data)  # Assuming 'data' is your original DataFrame
    
    # Get model config
    config = ModelNameConfig()
    
    # Call train_model with data subsets and config
    trained_model = train_model(x_train, x_test, y_train, y_test, config) 
    
    a, b = evaluation(trained_model, x_test, y_test)  
