# Fashion MNIST End-to-End MLOps Pipeline on Google Cloud Platform (GCP)

An end-to-end automated machine learning pipeline designed to train, register, and deploy a Convolutional Neural Network (CNN) for the Fashion MNIST dataset on Google Cloud Platform (GCP). This repository demonstrates standard MLOps practices utilizing Kubeflow Pipelines (KFP), Docker, and Agent Platform (Vertex AI).

## System Architecture

The workflow is orchestrated using a Google Cloud Service Account with restricted permissions. The architecture consists of three main stages:

1. Data Pipeline & Training (Kubeflow Pipelines):
   - Automates data ingestion from Cloud Storage.
   - Reshapes and standardizes pixel values to grayscale ($28 \times 28 \times 1$).
   - Trains a CNN architecture using TensorFlow.

2. Model Registry (Agent Platform):
   - Manages model lineage and version control.
   - Automatically registers production-ready `.keras` artifacts.

3. Online Prediction (Agent Platform Endpoints):
   - Packages the prediction service inside a custom FastAPI Docker container.
   - Deploys the container to a live Agent Platform Endpoint.
   - Serves raw 10-class probabilities to the client for inference and confidence calculations.

## Tech Stack & Tools

- **Machine Learning Framework**: TensorFlow (Keras)
- **API & Serving:** FastAPI, Uvicorn
- **Pipeline Orchestration**: Kubeflow Pipelines (KFP v2)
- **Containerization**: Docker, Artifact Registry
- **Cloud Infrastructure (GCP)**: Agent Platform (Pipelines, Model Registry, Endpoints), Cloud Storage, Cloud IAM

## Getting Started

0. Prerequisites
Before running any scripts, install the Google Cloud CLI and authenticate your local environment with your GCP account:

    ```bash
    # Authenticate your Google Cloud account
    gcloud auth login

    # Set up Application Default Credentials (ADC) for Python scripts
    gcloud auth application-default login

    # Set your active GCP project
    gcloud config set project <PROJECT_ID>
    ```

1. Compile the Pipeline Blueprint
Compile the end-to-end orchestrated pipeline into a deployable YAML configuration file:

    ```bash
    python pipeline.py
    ```

2. Build and Push the Serving Container
Rebuild the Docker image and push it to Google Artifact Registry whenever the FastAPI prediction logic is updated:

    ```bash
    # Define your cloud variables
    REGISTRY_URL="asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY_NAME>"
    IMAGE_NAME="<IMAGE_NAME>"
    TAG="<VERSION>"

    # Build the image
    docker build -t $REGISTRY_URL/$IMAGE_NAME:$TAG .

    # Push to Google Cloud Artifact Registry
    docker push $REGISTRY_URL/$IMAGE_NAME:$TAG
    ```

3. Deploy to Agent Platform Endpoint
Execute your deployment script to initialize the managed endpoint, pull the latest container image, and expose the URL for real-time client inferences.
    ```bash
    # Run the pipeline deployment:
    python deploy_pipeline.py

    # Run the endpoint deployment to expose the serving URL for real-time client inferences:
    python deploy_endpoint.py
    ```

## Cloud Security & IAM Configuration
For production-grade security, the automation pipeline operates using a dedicated Service Account configured with the following granular roles rather than broad primitive permissions:
- **Vertex AI User**: To execute orchestrations, manage model metadata, and provision endpoints.
- **Storage Object Admin**: To read ingested training datasets and commit trained model artifacts.
- **Artifact Registry Reader**: To pull the production-ready FastAPI serving images onto Vertex AI nodes.