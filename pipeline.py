import os
from kfp import dsl, compiler
from kfp.dsl import Output, Model


# =====================================================================
# Component 1: Data Preprocess
# =====================================================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["tensorflow", "numpy", "google-cloud-storage"]
)
def download_and_preprocess(bucket_name: str, data_path: str):
    import tensorflow as tf
    import numpy as np
    from google.cloud import storage
    import os
    
    print("--- Data Download and Preprocessing ---")

    # Load data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Temporary stored data in the container
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    # Send the saved data to the Cloud Storage Bucket
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for filename in ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]:
        blob = bucket.blob(os.path.join(data_path, filename))
        blob.upload_from_filename(filename)
    
    print("Data preparation completed and uploaded to GCS!")


# =====================================================================
# Component 2: Model Training
# =====================================================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["tensorflow", "numpy", "google-cloud-storage"]
)
def train_fashion_mnist(
    bucket_name: str, 
    data_path: str, 
    model_output_uri: Output[Model]
):
    import tensorflow as tf
    import numpy as np
    from google.cloud import storage
    import os

    # Access the stored data in bucket

    # Connect to the Google Cloud System (GCS)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for filename in ["X_train.npy", "y_train.npy"]:
        blob = bucket.blob(os.path.join(data_path, filename))
        blob.download_to_filename(filename)
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")

    # Create a model (same as the local version)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(X_train, y_train, epochs=3)

    os.makedirs(model_output_uri.path, exist_ok=True)
    model.save(os.path.join(model_output_uri.path, "model.keras"))
    print("Model trained and output artifact paths registered!")


# =====================================================================
# Component 3: Automate model registration with `.keras` support
# =====================================================================
@dsl.component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def upload_model_to_registry(
    project: str,
    location: str,
    display_name: str,
    model_artifact: dsl.Input[dsl.Model]
):
    from google.cloud import aiplatform
    from google.cloud.aiplatform import Model

    print(f"Initializing AI Platform for project: {project} at {location}")
    aiplatform.init(project=project, location=location)

    print("Checking for existing model versions in Model Registry...")
    gcs_model_uri = model_artifact.uri
    model_list = Model.list(filter=f'display_name="{display_name}"', order_by="create_time desc")
    parent_model = None
    if model_list:
        parent_model = model_list[0].resource_name
        print(f"Found existing model. Registering as a new version under parent: {parent_model}")
    else:
        print("No existing model found with this display name. Creating a new parent mode asset.")

    print(f"Automated uploading model from GCS Path: {gcs_model_uri}")
    
    uploaded_model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=gcs_model_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/custom-container-predict:latest",
        parent_model=parent_model
    )
    print(f"Successfully registered model! Resource Name: {uploaded_model.resource_name}")


# =====================================================================
# Pipeline Blueprint:
# =====================================================================
@dsl.pipeline(
    name="fashion-mnist-pipeline",
    description="An end-to-end pipeline with Model Registry incorporation"
)
def fashion_mnist_pipeline(bucket_name: str, data_path: str):
    # Component 1
    preprocess_task = download_and_preprocess(bucket_name=bucket_name, data_path=data_path)

    # Component 2
    train_task = train_fashion_mnist(
        bucket_name=bucket_name,
        data_path=data_path
    )
    # [download_and_preprocess] -> [train_fasion_mnist]
    train_task.after(preprocess_task)

    # Component 3
    model_upload_task = upload_model_to_registry(
        project="fashion-mnist-cloud",
        location="asia-southeast1",
        display_name="fashion-mnist-model-registry",
        model_artifact=train_task.outputs["model_output_uri"] 
    )
    # [download_and_preprocess] -> [train_fasion_mnist] -> [upload_model_to_registry]
    model_upload_task.after(train_task)


if __name__ == "__main__":
    print("--- Ready to compile blueprint! ---")
    compiler.Compiler().compile(
        pipeline_func=fashion_mnist_pipeline,
        package_path="fashion_mnist_pipeline.yaml"
    )
    print("Compiled blueprint successfully at 'fashion_mnist_pipeline.yaml'")