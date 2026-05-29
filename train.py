import os
import numpy as np
import tensorflow as tf


def download_and_preprocess_data():
    print("--- Data Download and Preprocessing ---")

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    print(f"Data loaded: Train shape = {X_train.shape}, Test shape = {X_test.shape}")

    return X_train, y_train, X_test, y_test


def train_fashion_mnist_model(X_train, y_train, X_test, y_test):
    print("--- Model Architecture and Training ---")

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

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("Training Completed!")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # `.keras` ดีกว่า `.h5` ทั้งในด้านความปลอดภัย / รองรับการ save และ load `Custom Layer` / ใช้กับ PyTorch หรือ JAX ได้ 
    model_save_path = "model/my_fashion_model.keras"
    model.save(model_save_path)
    print(f"Model saved locally at: {os.path.abspath(model_save_path)}")

    return model


if __name__ == "__main__":
    print("RUNNN")
    X_train, y_train, X_test, y_test = download_and_preprocess_data()
    fashion_model = train_fashion_mnist_model(X_train, y_train, X_test, y_test)