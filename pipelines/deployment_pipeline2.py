import json
import os
import numpy as np
import pandas as pd
from zenml import pipeline, step, get_step_context
from zenml.client import Client
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.steps import BaseParameters, Output
from mlflow.tracking import MlflowClient, artifact_utils
import mlflow
import mlflow.sklearn
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentConfig #just added




from steps.clean_data import clean_data
from steps.evaluation3 import evaluation
from steps.ingest_data import ingest_data
from steps.model_train3 import train_model
from materializer.custom_materializer import cs_materializer
from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=["mlflow"])

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0.0

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""
    return accuracy > config.min_accuracy

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline."""
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""
    service.start(timeout=10)  # should be a NOP if already started
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
    prediction = service.predict(data)
    return prediction

@step
def deploy_model(pipeline_name: str, model_uri: str) -> MLFlowDeploymentService:
    """Deploy a model using the MLflow Model Deployer"""
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="mlflow-model-deployment-example",
        description="An example of deploying a model using the MLflow Model Deployer",
        pipeline_name=pipeline_name,  # Pass the pipeline name here
        model_uri=model_uri,
        model_name="model",
        workers=1,
        mlserver=False,
        timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT,
    )
    
    service = model_deployer.deploy_model(mlflow_deployment_config, service_type=MLFlowDeploymentService.SERVICE_TYPE)
    import pdb; pdb.set_trace()
    return service

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    pipeline_name="continuous_deployment_pipeline"
    # Link all the steps artifacts together
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)
    deployment_decision = deployment_trigger(accuracy=mse)

    if deployment_decision:
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "model")
            model_uri = f"runs:/{run.info.run_id}/model"
            print(f"Model logged to: {model_uri}")
            
            # Deploy the model using the deploy_model step
            deploy_model(pipeline_name=pipeline_name,model_uri=model_uri)

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)