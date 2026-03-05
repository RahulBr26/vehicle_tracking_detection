from ultralytics import YOLO
import cv2
from collections import defaultdict

# Load YOLO model
model = YOLO("yolov8n.pt")
class_names = model.names

# Open video
cap = cv2.VideoCapture("traffic.mp4")

# Line positions
line_y_red = 198
line_y_green = line_y_red + 100

# Counting dictionaries
counted_ids_red = defaultdict(int)
counted_ids_green = defaultdict(int)

# Track crossing state
crossed_red_first = {}
crossed_green_first = {}

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    # Get frame width automatically
    height, width, _ = frame.shape

    # Run YOLO tracking
    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.cpu().int().tolist()
        class_indices = results[0].boxes.cls.cpu().int().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Draw lines
        cv2.line(frame, (0, line_y_red), (width, line_y_red), (0,0,255), 3)
        cv2.putText(frame,"Red Line",(10,line_y_red-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        cv2.line(frame, (0, line_y_green), (width, line_y_green), (0,255,0), 3)
        cv2.putText(frame,"Green Line",(10,line_y_green-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):

            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_idx]

            # Draw bounding box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(frame,
                        f"{class_name} ID:{track_id} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        2)

            # Calculate center
            center_y = int((y1 + y2) / 2)

            # RED LINE CROSSING
            if center_y < line_y_red and track_id not in crossed_red_first:
                crossed_red_first[track_id] = True

            if center_y > line_y_red and track_id in crossed_red_first:
                counted_ids_red[class_name] += 1
                del crossed_red_first[track_id]

            # GREEN LINE CROSSING
            if center_y < line_y_green and track_id not in crossed_green_first:
                crossed_green_first[track_id] = True

            if center_y > line_y_green and track_id in crossed_green_first:
                counted_ids_green[class_name] += 1
                del crossed_green_first[track_id]

    # Show counts
    y_offset = 30

    cv2.putText(frame,"RED LINE COUNT",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    for cls, count in counted_ids_red.items():
        cv2.putText(frame,f"{cls}: {count}",
                    (10,y_offset+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,255),
                    2)
        y_offset += 25

    y_offset = 30

    cv2.putText(frame,"GREEN LINE COUNT",(250,20),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    for cls, count in counted_ids_green.items():
        cv2.putText(frame,f"{cls}: {count}",
                    (250,y_offset+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)
        y_offset += 25

    # Show frame
    cv2.imshow("YOLOv8 Traffic Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()