from ultralytics import YOLO
import cv2

# ---- YOLO ORIGINAL MODEL ---- #
# model = YOLO("yolo11n.pt")

# ---- YOLO FINE-TUNED MODEL ---- #
model = YOLO('D:/CE PROJECT/runs/detect/train4/weights/best.pt')

# ---- RESULTS ---- #
def classification(image_path, save_image=True, save_path="results/cat_detected_nose1.jpg"):
    results = model(image_path)
    image = cv2.imread(image_path)

    # if save_image:
    #     results[0].save(filename=save_path)

    detections = []

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cropped = image[y1:y2, x1:x2]

            detection_data = {
                "class": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2],
                "image": cropped
            }
            detections.append(detection_data)

    return detections

# ---- TEST ---- #
# image_path = "images/cat_test1.jpg"
# detections = classification(image_path)

# for data in detections:
#     if data["class"] in ["dog", "cat"]:
#         print(f"Detected : {data['class']} ({data['confidence']:.2f}) at {data['box']}")
#         cv2.imshow("Image", data['image'])