import os
from google.cloud import aiplatform


def trigger_vertex_pipeline():
    print("\n--- [1/3] Initializing Connection to Google Cloud Platform ---")

    PROJECT_ID = "fashion-mnist-cloud"
    REGION = "asia-southeast1"
    BUCKET_NAME = "fon-fashion-mnist-bucket"

    aiplatform.init(project=PROJECT_ID, location=REGION)    
    print(f"Connected to Project: {PROJECT_ID} at region: {REGION}")

    print("\n--- [2/3] Preparing Pipeline Job from Blueprint ---")
    
    pipeline_job = aiplatform.PipelineJob(
        display_name="fashion_mnist_vertex_run",
        template_path="fashion_mnist_pipeline.yaml",
        pipeline_root=f"gs://{BUCKET_NAME}/vertex_pipeline_root",
        parameter_values={
            "bucket_name": BUCKET_NAME,
            "data_path": "fashion_mnist_data"
        },
        enable_caching=False
    )

    print("Pipeline job configuration loaded successfully.")
    print("\n--- [3/3] Launching Pipeline Blueprint to Vertex AI ---")

    pipeline_job.run()

    print("\nMission Launched! The pipeline is now running serverless on Google Cloud.")
    print("You can close this terminal and monitor the progress via the Vertex AI Web Console.")


if __name__ == "__main__":
    trigger_vertex_pipeline()
