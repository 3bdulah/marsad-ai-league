import cv2
import numpy as np

# Draw a bounding box with a label
def draw_box(img, box, label="Object", color=(0, 255, 0)):
    x, y, w, h = box
    top_left = (int(x), int(y))
    bottom_right = (int(x + w), int(y + h))
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    cv2.putText(img, label, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Simulated alert function
def trigger_alert(message):
    print(f"🚨 ALERT: {message}")

# Normalize bounding box
def normalize_box(x, y, w, h, img_w, img_h):
    return x / img_w, y / img_h, w / img_w, h / img_h

# Denormalize box (YOLO format to pixel coords)
def denormalize_box(x, y, w, h, img_w, img_h):
    return x * img_w, y * img_h, w * img_w, h * img_h

# Convert YOLO label to (x, y, w, h)
def yolo_label_to_box(line, img_shape):
    parts = line.strip().split()
    x, y, w, h = map(float, parts[1:])
    img_h, img_w = img_shape[:2]
    return denormalize_box(x, y, w, h, img_w, img_h)

# Overlay text on image
def overlay_text(img, text, pos=(10, 30), color=(0, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)