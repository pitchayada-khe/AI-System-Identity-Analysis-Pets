from final_model import classification
import cv2

if __name__ == "__main__":
    image_path = "images/dog_test.jpg"
    detections = classification(image_path)

    for data in detections:
        if data["class"].lower() in ["dog", "cat"]:
            print(f"Animal Type Detected : {data['class']} ({data['confidence']:.2f}) at {data['box']}")
            cv2.imshow("Animal", data["image"])
            cv2.imwrite(f"results/cropped/{data['class']}_animal.jpg", data["image"])
            cv2.waitKey(0)

        if data["class"].lower() == "nose":
            print(f"Nose Detected : {data['class']} ({data['confidence']:.2f}) at {data['box']}")
            cv2.imshow("Nose", data["image"])
            cv2.imwrite("results/cropped/{data['class']}_nose.jpg", data["image"])
            cv2.waitKey(0)

cv2.destroyAllWindows()