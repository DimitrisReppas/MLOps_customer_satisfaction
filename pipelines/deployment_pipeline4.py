import json
import os
from typing import Any

import numpy as np
import pandas as pd
import mlflow

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.steps import BaseParameters
from steps.clean_data import clean_data
from steps.evaluation3 import evaluation
from steps.ingest_data import ingest_data
from steps.model_train3 import train_model
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=["mlflow"])

@step
def mlflow_model_deployer_step(model: Any, deploy_decision: bool, port: int = 5000) -> None:
    """Serve the model using MLflow's local server functionality."""
    if deploy_decision:
        # Save the model to a temporary directory
        temp_model_path = "model_temp"
        mlflow.pyfunc.save_model(path=temp_model_path, python_model=model)

        # Serve the model using MLflow
        os.system(f"mlflow models serve -m {temp_model_path} -p {port}")
        print(f"Model is being served at http://127.0.0.1:{port}")
    else:
        print("Deployment decision was False. Model not deployed.")

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0.0

@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""
    return accuracy > config.min_accuracy

@step(enable_cache=False)
def prediction_service_loader(port: int = 5000) -> str:
    """Returns the URL of the locally served MLFlow model."""
    model_server_url = f"http://127.0.0.1:{port}/invocations"
    return model_server_url

@step
def predictor(service_url: str, data: str) -> np.ndarray:
    """Run an inference request against the local prediction service"""
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    
    import requests
    response = requests.post(service_url, json={"dataframe_split": data})
    prediction = np.array(response.json()["predictions"])
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(min_accuracy: float = 0.92, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=mse, config=DeploymentTriggerConfig(min_accuracy=min_accuracy))
    mlflow_model_deployer_step(model=model, deploy_decision=deployment_decision, port=5000)

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    batch_data = dynamic_importer()
    model_server_url = prediction_service_loader()
    predictor(service_url=model_server_url, data=batch_data)
