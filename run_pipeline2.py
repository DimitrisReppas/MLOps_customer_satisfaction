from pipelines.training_pipeline import train_pipeline
from steps.clean_data import clean_data
from steps.evaluation3 import evaluation
from steps.ingest_data import ingest_data
from steps.model_train3 import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from steps.config import ModelNameConfig 

if __name__ == "__main__":
    print("Starting pipeline...")
    config = ModelNameConfig()
    df = ingest_data()
    print("Ingested data:", df)

    x_train, x_test, y_train, y_test = clean_data(df)
    print("Cleaned data:", x_train, x_test, y_train, y_test)

    model = train_model(x_train, x_test, y_train, y_test, config)
    print("Trained model:", model)

    mse, rmse = evaluation(model, x_test, y_test)
    print("Evaluation results:", mse, rmse)

    training = train_pipeline(
        ingest_data=ingest_data,
        clean_data=clean_data,
        model_train=train_model,
        evaluation=evaluation,
       # stack_name='new_stack'
    )
    
    import pdb; pdb.set_trace()
    training.run()
    
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )