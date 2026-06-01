import numpy as np
from PIL import Image
from google.cloud import aiplatform

# ==========================================
# 1. Configuration
# ==========================================
REGION = "asia-southeast1"
PROJECT_ID = "fashion-mnist-cloud"            
ENDPOINT_ID = "6012923428074422272"

# Fashion-MNIST class labels
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

TEST_IMG_PATH = "images/travel-bag-canvas-green-beige.jpg"

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
    response = endpoint.predict(instances=instances)
    print(f"Response: {response}")

    # Extract prediction probabilities
    predictions = response.predictions[0]
    
    # Identify the class with the highest probability
    predicted_class_idx = np.argmax(predictions)
    print(predicted_class_idx)
    predicted_label = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100

    print("\n--- [ Prediction Results ] ---")
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print("------------------------------")

if __name__ == "__main__":
    predict_fashion()