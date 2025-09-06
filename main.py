from final_model import classification
import cv2
from pathlib import Path
import os

if __name__ == "__main__":
    image_path = "dogs_cats_dataset/Cat/Cat_Bengalthor/Cat_Bengalthor_18.jpg"
    detections = classification(image_path)

    if detections is None:
        print(f"{image_path} - Not found Dog or Cat!")
    else:
        print(f"Animal Detected: {detections['class']} ({detections['confidence']:.2f})")

        face_dir = Path("results/Test_Progress_3/Cats/Cat_Bengalthor/Face")
        nose_dir = Path("results/Test_Progress_3/Cats/Cat_Bengalthor/Nose")
        face_dir.mkdir(parents=True, exist_ok=True)
        nose_dir.mkdir(parents=True, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        face_path = face_dir / f"{base_name}_face.jpg"
        cv2.imwrite(str(face_path), detections["image"])
        cv2.imshow("Face", detections["image"])

        if detections["nose_data"]:
            print(f"Nose Detected: {detections['nose_data']['class']} ({detections['nose_data']['confidence']:.2f})")

            nose_path = nose_dir / f"{base_name}_nose.jpg"
            cv2.imwrite(str(nose_path), detections["nose_data"]["image"])
            cv2.imshow("Nose", detections["nose_data"]["image"])
        else:
            print(f"{image_path} - Not found Nose!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
