from ultralytics import YOLO
import cv2
from pathlib import Path
from datetime import datetime

# ---- YOLO PRE-TRAIN MODEL ---- #
animal_model = YOLO("yolo11n.pt")

# ---- YOLO FINE-TUNED MODEL ---- #
nose_model = YOLO('D:/CE PROJECT/runs/detect/train4/weights/best.pt')

# ---- PROCESS ---- #
def classification(image_path, save_image=True, save_dir="results/", resize_dim=(128,128)):
    # Path(save_dir).mkdir(parents=True, exist_ok=True)
    image = cv2.imread(image_path)

    # DETECT ANIMAL #
    results_animal = animal_model(image_path)
    best_animal = None
    best_conf = -1
    best_box = None

    for result in results_animal:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = animal_model.names[class_id]

            if class_name not in ["dog", "cat"]:
                continue

            animal_conf = float(box.conf[0])
            if animal_conf > best_conf:
                best_conf = animal_conf
                best_animal = class_name
                best_box = map(int, box.xyxy[0])

    if best_animal is None:
        return None
    
    x1, y1, x2, y2 = best_box
    face = image[y1:y2, x1:x2]

    face_region = cv2.resize(face, resize_dim, interpolation=cv2.INTER_LINEAR)

    detection_data = {
        "class": best_animal,
        "confidence": best_conf,
        "image": face_region,
        "nose_data": None
    }

    # DETECT NOSE FROM FACE REGION #
    results_nose = nose_model(face_region)
    best_nose = None
    best_nose_conf = -1

    for nose_result in results_nose:
        for nose_box in nose_result.boxes:
            class_id = int(nose_box.cls[0])
            class_name = nose_model.names[class_id]

            if class_name != "nose":
                continue

            nose_conf = float(nose_box.conf[0])
            if nose_conf > best_nose_conf:
                best_nose_conf = nose_conf
                nx1, ny1, nx2, ny2 = map(int, nose_box.xyxy[0])
                nose = face_region[ny1:ny2, nx1:nx2]
                nose_img = cv2.resize(nose, resize_dim, interpolation=cv2.INTER_LINEAR)

                best_nose = {
                    "class": "nose",
                    "confidence": nose_conf,
                    "image": nose_img
                }

    if best_nose:
        detection_data["nose_data"] = best_nose

    # if save_image and results_animal:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     annotated_img = results_animal[0].plot()
    #     cv2.imwrite(f"{save_dir}{timestamp}_annotated.jpg", annotated_img)
    #     cv2.imwrite(f"{save_dir}{timestamp}_{best_animal}_face.jpg", face_region)
        
    #     if best_nose:
    #         cv2.imwrite(f"{save_dir}{timestamp}_{best_animal}_nose.jpg", best_nose["image"])

    return detection_data

# ---- TEST ---- #
# test_image_path = "results/pred_yolov11_best/dog_cat_test1.jpg"
# save_output_path = "results/test_max_conf/"

# result = classification(test_image_path, save_image=True, save_dir=save_output_path)

# if result is None:
#     print("Not found Dog or Cat!")
# else:
#     print(f"Class : {result['class']}, Confidence : {result['confidence']:.2f}")

#     cv2.imshow("Face Region", result["image"])

#     if result["nose_data"]:
#         print(f"Class : {result['nose_data']['class']}, Confidence : {result['nose_data']['confidence']:.2f}")
#         cv2.imshow("Nose Region", result["nose_data"]["image"])
#     else:
#         print("Not found Nose!")

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()