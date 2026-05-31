import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
from typing import List
from google.cloud import storage


app = FastAPI()


class PredictionInput(BaseModel):
    # Image Shape: [batch_size, 28, 28, 1]
    instances: List[List[List[List[float]]]]


MODEL_DIR = os.environ.get("AIP_STORAGE_URI")
local_model_path = "/tmp/model.keras"

print(f"Downloading model from GCS: {MODEL_DIR}")
bucket_name = MODEL_DIR.replace("gs://", "").split("/")[0]
print(f"Bucket name: {bucket_name}")
blob_name = "/".join(MODEL_DIR.replace("gs://", "").split("/")[1:]) + "/model.keras"
print(f"Blob name: {blob_name}")

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_name)
blob.download_to_filename(local_model_path)

model = tf.keras.models.load_model(local_model_path)
print("Model loaded successfully into FastAPI container!")


@app.get("/health")
def health_check():
    return {"status": "health"}

@app.post("/predict")
def predict(data: PredictionInput):
    inputs = np.array(data.instances, dtype=np.float32)
    predictions = model.predict(inputs)
    predicted_classes = np.argmax(predictions, axis=1).tolist()

    return {"predictions": predicted_classes}