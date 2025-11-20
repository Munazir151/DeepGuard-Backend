from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")    # nano model → faster for real-time

# Input video
cap = cv2.VideoCapture("2.mp4")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform tracking
    results = model.track(frame, persist=True)

    # YOLO returns an annotated frame automatically
    output_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLOv8 Object Tracking Output", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
