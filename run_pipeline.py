from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation3 import evaluation
from steps.ingest_data import ingest_data
from steps.model_train3 import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":
    training = train_pipeline(
        ingest_data(),
        clean_data(),
        train_model(),
        evaluation(),
    )
    import pdb; pdb.set_trace()
    training.run()
    
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )
