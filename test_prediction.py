import numpy as np
from PIL import Image
from google.cloud import aiplatform

# ==========================================
# 1. Configuration
# ==========================================
REGION = "asia-southeast1"
PROJECT_ID = "fashion-mnist-cloud"            
ENDPOINT_ID = "336699047727333376"

# Fashion-MNIST class labels
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

TEST_IMG_PATH = "images/shoes1.jpg"

# ==========================================
# 2. Image Preprocessing
# ==========================================
def preprocess_image(image_path):
    # Convert image to grayscale and resize to 28x28 pixels
    img = Image.open(image_path).convert('L').resize((28, 28))
    
    # Convert to NumPy array and normalize pixel values to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch and channel dimensions to match model input shape (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1)) 
    
    # Convert to standard Python list for JSON serialization
    return img_array.tolist()

# ==========================================
# 3. Vertex AI Prediction Request
# ==========================================
def predict_fashion():
    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Reference the deployed endpoint
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
    
    # Prepare input instances
    instances = preprocess_image(TEST_IMG_PATH)
    
    print("Sending prediction request to Vertex AI Endpoint...")
    print("Sending prediction request to Vertex AI Endpoint...")
    response = endpoint.predict(instances=instances)
    
    # Since the response predictions is just a single float, we handle it directly
    confidence_value = response.predictions[0]
    
    confidence = confidence_value * 100 if confidence_value <= 1.0 else confidence_value

    print("\n--- [ Prediction Results ] ---")
    print(f"Confidence Score: {confidence:.2f}%")
    print("------------------------------")

if __name__ == "__main__":
    predict_fashion()