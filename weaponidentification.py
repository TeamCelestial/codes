import cv2
from ultralytics import YOLO

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the webcam frame
    results = model(frame)

    # Annotate the frame with the detected objects
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8 Weapon Detection", annotated_frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
