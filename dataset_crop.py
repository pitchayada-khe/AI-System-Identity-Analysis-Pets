from final_model import classification
from pathlib import Path
import os
import cv2

input_folder = Path("dogs_cats_dataset/Dog/Dog_Suihanki")
output_base = Path("results/Test_Progress_3/Dogs/Dog_Suihanki")

for image_path in input_folder.glob("*.[jp][pn]g"):
    print(f"\n---------------------------------\nProcessing file: {image_path.name}")
    detections = classification(str(image_path))

    if detections is None:
        print(f"Not found Dog or Cat!")
        continue

    print(f"Animal Detected: {detections['class']} ({detections['confidence']:.2f})")

    face_dir = output_base / "Face"
    nose_dir = output_base / "Nose"
    face_dir.mkdir(parents=True, exist_ok=True)
    nose_dir.mkdir(parents=True, exist_ok=True)

    base_name = os.path.splitext(image_path.name)[0]

    face_path = face_dir / f"{base_name}_face.jpg"
    cv2.imwrite(str(face_path), detections["image"])

    if detections["nose_data"]:
        print(f"Nose Detected: {detections['nose_data']['class']} ({detections['nose_data']['confidence']:.2f})")
        nose_path = nose_dir / f"{base_name}_nose.jpg"
        cv2.imwrite(str(nose_path), detections["nose_data"]["image"])
    else:
        print(f"Not found Nose!")
