from ultralytics import YOLO
import os

# Dataset and config
DATA_CONFIG = "../data.yaml"  # Define class names and paths
MODEL_NAME = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 640

def train():
    print("🚀 Starting YOLOv8 Training for Crowd Safety Detection...")

    # Load base model
    model = YOLO(MODEL_NAME)

    # Train on custom data
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        name="marsad_crowd_model",
        project="../models"
    )

    print("✅ Training completed. Model saved to /models/marsad_crowd_model/")

if __name__ == "__main__":
    train()