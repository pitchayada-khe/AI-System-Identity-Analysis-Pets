from ultralytics import YOLO
import cv2

# ---- YOLO ORIGINAL MODEL ---- #
model = YOLO("yolo11n.pt")

# ---- DETECT FROM IMAGE ---- #
results = model("images/dog_test.jpg")

# ---- RESULTS ---- #
results[0].show()
# results[0].save(filename="results/dog_detected.jpg")

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"Detected : {class_name} ({confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

# ---- DETECT FROM CAMERA OR VIDEO ---- #
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     cv2.imshow("YOLO Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()