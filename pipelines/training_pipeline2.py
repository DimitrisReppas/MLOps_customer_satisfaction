import sys
import os
# Append the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(ingest_data, clean_data, model_train, evaluation):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train(x_train, x_test, y_train, y_test) # ego vlepo auto mallon lathos
    mse, rmse = evaluation(model, x_test, y_test)
    
    
if __name__ == "__main__":
    from steps.ingest_data import ingest_data
    from steps.clean_data import clean_data
    from steps.model_train3 import * 
    from steps.evaluation3 import *
    # Assuming ingest_data, clean_data, model_train, and evaluation are defined somewhere
    # You need to provide these functions as arguments to train_pipeline.execute()
    train_pipeline.execute(ingest_data, clean_data, model_train3, evaluation)