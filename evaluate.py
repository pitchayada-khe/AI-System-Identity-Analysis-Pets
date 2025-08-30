from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
import glob

# ---- EVALUATE ---- #
def evaluate_accuracy(model_path, data, imgsz=640, batch=16, device=0):
    model = YOLO(model_path)
    metrics = model.val(data=data, imgsz=imgsz, batch=batch, device=device, verbose=False)
    
    results = {
        "mAP50_95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "Precision": float(metrics.box.p.mean()),
        "Recall": float(metrics.box.r.mean()),
        "F1": float(metrics.box.f1.mean())         
    }
    return model, results

def evaluate_speed(model, images, imgsz=640, device=0):
    for _ in range(5):
        model.predict(source=images[0], imgsz=imgsz, device=device, verbose=False)

    start = time.time()
    n = 0
    for _ in model.predict(source=images, imgsz=imgsz, device=device, stream=True, verbose=False, save=True, project="results", name="predict_evaluate"):
        n += 1
    end = time.time()

    total_time = end - start
    latency = (total_time / n) * 1000 
    fps = n / total_time
    return {"Latency(ms/img)": latency, "FPS": fps}

# ---- RESULTS ---- #
def plot_results(df, save_path=None):
    x = np.arange(len(df["Model"]))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8,5))

    ax1.bar(x - width, df["mAP50_95"], width, label="mAP@0.5:0.95")
    ax1.bar(x, df["Precision"], width, label="Precision")
    ax1.bar(x + width, df["Recall"], width, label="Recall")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"])

    ax2 = ax1.twinx()
    ax2.plot(x, df["FPS"], marker="o", color="red", label="FPS")
    ax2.set_ylabel("FPS")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("YOLOv11_12 : Accuracy And Speed")

    if save_path:
        plt.savefig(save_path)

    plt.show()

if __name__ == "__main__":
    model_a = "D:/CE PROJECT/runs/detect/train4/weights/last.pt"
    model_b = "D:/CE PROJECT/runs/detect/train5/weights/best.pt"
    data_yaml = "D:/CE PROJECT/models/dataset2/data.yaml"

    model_a, acc_a = evaluate_accuracy(model_a, data_yaml)
    model_b, acc_b = evaluate_accuracy(model_b, data_yaml)

    test_images = glob.glob("images/*.*")
    speed_a = evaluate_speed(model_a, test_images)
    speed_b = evaluate_speed(model_b, test_images)

    df = pd.DataFrame([
        {"Model": "YOLOv11-last", **acc_a, **speed_a},
        {"Model": "YOLOv12-best", **acc_b, **speed_b}
    ])

    print(df)
    df.to_csv("evaluate_yolov11_12.csv", index=False)
    plot_results(df, save_path="evaluate_yolov11_12.png")
