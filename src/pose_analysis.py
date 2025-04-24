import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# Load sample video
video_path = "../dataset/player_video.mp4"  # Replace with a real or placeholder video
cap = cv2.VideoCapture(video_path)

print("🎯 Starting player pose analysis...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get pose landmarks
    results = pose.process(frame_rgb)

    # Draw pose on frame
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Player Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()