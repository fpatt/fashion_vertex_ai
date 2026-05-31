from google.cloud import aiplatform

# Initialize Vertex AI SDK with project details
aiplatform.init(
    project="fashion-mnist-cloud", 
    location="asia-southeast1"
)

# GCS URI where the trained Keras model artifact is stored
gcs_path = "gs://fon-fashion-mnist-bucket/vertex_pipeline_root/227982367383/fashion-mnist-pipeline-20260531005036/train-fashion-mnist_1916572187393785856/model_output_uri"

print("Executing workaround: Switching to a custom container to bypass .pb file constraints for .keras file")

# Register the model using a generic custom container to bypass the strict TensorFlow pre-built container validation
model = aiplatform.Model.upload(
    display_name="fashion-mnist-model-registry",
    artifact_uri=gcs_path,
    # Using a generic prediction image prevents Vertex AI from enforcing the traditional `saved_model.pb` structure
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/custom-container-predict:latest"
)

print(f"The model has been successfully registered in the Model Registry: {model.resource_name}")