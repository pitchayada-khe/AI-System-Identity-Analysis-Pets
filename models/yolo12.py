from ultralytics import YOLO
import torch

def main():
    
    print("Is CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))

    # ---- YOLO ORIGINAL MODEL ---- #
    model_v2 = YOLO("yolo12n.pt")

    # ---- TRAIN SECTION ---- #
    model_v2.train(
        data='D:/CE PROJECT/models/dataset2/data.yaml',
        epochs=50,
        imgsz=320,
        device=0
    )

    # ---- RESULTS ---- #
    metrics = model_v2.val()
    print(metrics)

if __name__ == "__main__":
    main()