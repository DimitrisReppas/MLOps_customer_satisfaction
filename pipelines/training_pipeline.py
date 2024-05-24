from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
import sys
import os

# Add the parent directory (your_project) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now you can import the module
from steps.config import ModelNameConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])

#config = ModelNameConfig()
#import pdb; pdb.set_trace()
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
     # Get model config
    config = ModelNameConfig()
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train(x_train, x_test, y_train, y_test, config)
    mse, rmse = evaluation(model, x_test, y_test)