from ultralytics import YOLO
import cv2

print("========== PROGRAM STARTED ==========")

try:
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("YOLO model loaded successfully!")
except Exception as e:
    print("Error loading YOLO model:", e)
    exit()

print("Opening Camera...")
cap = cv2.VideoCapture(0)

print("Camera Opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Camera could not be opened!")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from camera.")
        break

    results = model(frame)

    person_found = False

    for r in results:

        annotated = r.plot()

        for box in r.boxes:
            cls = int(box.cls[0])

            # COCO Class 0 = Person
            if cls == 0:
                person_found = True

    if person_found:
        cv2.putText(
            annotated,
            "PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            annotated,
            "NO PERSON",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("DRCC AI Detection", annotated)

    key = cv2.waitKey(1)

    if key == ord('q'):
        print("Closing...")
        break

cap.release()
cv2.destroyAllWindows()

print("Program Finished.")