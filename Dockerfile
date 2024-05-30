# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements2.txt



# Initialize ZenML repository
RUN zenml init

# Install MLflow integration
RUN zenml integration install mlflow -y

# Register MLflow experiment tracker
RUN zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register MLflow model deployer
RUN zenml model-deployer register mlflow --flavor=mlflow

# Run deployment script when the container launches
CMD ["python", "run_deployment.py"]