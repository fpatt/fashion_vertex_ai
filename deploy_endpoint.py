from google.cloud import aiplatform


# =====================================================================
# Configuration
# =====================================================================
REGION = "asia-southeast1"
PROJECT_ID = "fashion-mnist-cloud"
MODEL_DISPLAY_NAME = "fashion-mnist-model-registry"
ENDPOINT_DISPLAY_NAME = "fashion-mnist-endpoint-v5"

REPO_NAME = "fashion-mnist-repo"
IMAGE_NAME = "fastapi-predictor"
IMAGE_TAG = "v9"
SERVING_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}"


def deploy_model_to_endpoint():
    print("Initializing Vertex AI SDK...")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # 1. Fetch the latest model from Model Registry
    print(f"Searching for model: {MODEL_DISPLAY_NAME}...")
    models = aiplatform.Model.list(
        filter=f'display_name="{MODEL_DISPLAY_NAME}"',
        order_by="create_time desc"
    )
    if not models:
        raise RuntimeError(f"Model '{MODEL_DISPLAY_NAME}' not found in registry. Please run pipeline first!")
    
    target_model = models[0]
    print(f"Found Model! Resource Name: {target_model.resource_name}")

    # 2. Get existing Endpoint or create a new one
    print(f"Checking for existing Endpoint to clear: {ENDPOINT_DISPLAY_NAME}...")
    existing_endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_DISPLAY_NAME}"')
    
    for ep in existing_endpoints:
        print(f"Force deleting stuck endpoint: {ep.resource_name} (this will stop the spinning)...")
        try:
            ep.undeploy_all()
            ep.delete()
            print("Stuck endpoint deleted successfully!")
        except Exception as e:
            print(f"Note: Could not delete directly, forcing creation of a clean one. Error: {e}")

    print("Creating a brand new clean Endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
    print(f"New Endpoint created! Resource Name: {endpoint.resource_name}")

    # 3. Deploy the container to the endpoint
    print("Deploying model to endpoint (this might take 10-15 minutes)...")

    model_deployed = endpoint.deploy(
        model=target_model,
        deployed_model_display_name="fashion-mnist-v6-deployment",
        machine_type="e2-standard-2",
        min_replica_count=1,
        max_replica_count=1
    )
    
    print("\n=========================================================")
    print("DEPLOYMENT SUCCESSFUL!")
    print(f"Endpoint HTTP URI: {endpoint.resource_name}")
    print("Your FastAPI container is now live and ready for production requests!")
    print("=========================================================")


if __name__ == "__main__":
    deploy_model_to_endpoint()