from ultralytics import YOLO
import torch

def main():
    
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    # ---- YOLO ORIGINAL MODEL ---- #
    model = YOLO("yolo11n.pt")

    # ---- TRAIN SECTION ---- #
    model.train(
        data='D:/CE PROJECT/models/dataset2/data.yaml',
        epochs=50,
        imgsz=320,
        device=0
    )

    # ---- RESULTS ---- #
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()