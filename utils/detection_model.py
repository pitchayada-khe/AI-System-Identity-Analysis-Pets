from ultralytics import YOLO
import cv2
from pathlib import Path
from datetime import datetime

from huggingface_hub import hf_hub_download

# ---- YOLO FINE-TUNED MODEL ---- #
hf_hub_download(repo_id="Muyumq/Dog-Cat_Identification", filename="yolo_best.pt", local_dir="models/yolo")
detection_model = YOLO('models/yolo/yolo_best.pt')

# ---- BOUNDING BOX COLORS ---- #
CLASS_COLORS = {
    'dog': (204, 153, 255),   
    'cat': (255, 153, 204),    
    'nose': (170, 205, 102)
}
DEFAULT_COLOR = (250, 206, 135)
CONF_THRESHOLD = 0.7

# ---- PROCESS ---- #
def detection(frame, resize_dim=(224,224)):

    # For displaying in GUI (annotated)
    image = frame.copy()
    clean_frame = frame.copy()
    img_h, img_w = image.shape[:2]
    scale = max(img_w / 640.0, img_h / 480.0)
    dynamic_thickness = max(1, int(2 * scale))
    font_scale = max(0.4, 0.6 * scale)

    # DETECT ANIMAL #
    results_animal = detection_model(image)
    best_animal = None
    best_conf = -1
    best_box = None

    for result in results_animal:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]

            if class_name not in ["dog", "cat"]:
                continue

            animal_conf = float(box.conf[0])
            if animal_conf < CONF_THRESHOLD:
                continue

            if animal_conf > best_conf:
                best_conf = animal_conf
                best_animal = class_name
                best_box = list(map(int, box.xyxy[0]))

    if best_animal is None:
        return None
    
    x1, y1, x2, y2 = best_box
    
    # Draw on 'image' for GUI
    animal_color = CLASS_COLORS.get(best_animal, DEFAULT_COLOR) 
    # cv2.rectangle(image, (x1, y1), (x2, y2), animal_color, dynamic_thickness)
    # cv2.putText(image, best_animal.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, animal_color, dynamic_thickness)
    label = f"{best_animal.upper()}"
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, dynamic_thickness)
    cv2.rectangle(image, (x1, y1), (x1 + text_w + 6, y1 + text_h + 6), animal_color, -1)
    cv2.putText(image, label, (x1 + 3, y1 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), dynamic_thickness)
    cv2.rectangle(image, (x1, y1), (x2, y2), animal_color, dynamic_thickness)


    # Crop from clean frame
    face_clean = clean_frame[y1:y2, x1:x2]
    # Crop from annotated frame to search for nose in (though clean could also be used, clean is safer)
    face_for_nose_search = clean_frame[y1:y2, x1:x2]

    if face_clean.size == 0 or face_for_nose_search.size == 0:
        return None

    face_region = cv2.resize(face_clean, resize_dim, interpolation=cv2.INTER_CUBIC)

    detection_data = {
        "class": best_animal,
        "confidence": best_conf,
        "image": face_region,
        "annotated_frame": image,
        "nose_data": None
    }

    # DETECT NOSE FROM CLEAN FACE REGION #
    results_nose = detection_model(face_for_nose_search)
    best_nose = None
    best_nose_conf = -1

    for nose_result in results_nose:
        for nose_box in nose_result.boxes:
            class_id = int(nose_box.cls[0])
            class_name = detection_model.names[class_id]

            if class_name != "nose":
                continue

            nose_conf = float(nose_box.conf[0])
            if nose_conf > best_nose_conf:
                best_nose_conf = nose_conf
                nx1, ny1, nx2, ny2 = map(int, nose_box.xyxy[0])
                
                # The nose coordinates are relative to the face crop
                # Draw on the GUI image (need to offset by face coordinates x1, y1)
                global_nx1, global_ny1 = x1 + nx1, y1 + ny1
                global_nx2, global_ny2 = x1 + nx2, y1 + ny2
                nose_color = CLASS_COLORS.get("nose", DEFAULT_COLOR)
                cv2.rectangle(image, (global_nx1, global_ny1), (global_nx2, global_ny2), nose_color, dynamic_thickness)
                
                # Crop clean nose
                nose_clean = face_clean[ny1:ny2, nx1:nx2]
                if nose_clean.size == 0:
                    continue
                nose_img = cv2.resize(nose_clean, resize_dim, interpolation=cv2.INTER_CUBIC)

                best_nose = {
                    "class": "nose",
                    "confidence": nose_conf,
                    "image": nose_img
                }

    if best_nose is None:
        return None

    detection_data["nose_data"] = best_nose

    return detection_data