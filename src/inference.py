import cv2
from ultralytics import YOLO

# Load trained model
model_path = "../models/marsad_crowd_model/weights/best.pt"  # Simulated path
model = YOLO(model_path)

# Video input (can be a real .mp4 or a webcam stream)
video_path = "../dataset/sample_stadium.mp4"
cap = cv2.VideoCapture(video_path)

print("🎥 Running real-time crowd analysis...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()
    cv2.imshow("مَرْصَد - Stadium Monitoring", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()