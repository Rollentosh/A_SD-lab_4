from ultralytics import YOLO

def generate_metrics_report(weights_path, data_yaml):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, split="val")

    precision = metrics.box.mp       # Mean Precision
    recall = metrics.box.mr          # Mean Recall
    map50 = metrics.box.map50        # mAP@50
    map50_95 = metrics.box.map       # mAP@50-95

    report = f"""
    === ОТЧЁТ О КАЧЕСТВЕ МОДЕЛИ (YOLOv8s — Дорожные конусы) ===
    • Precision (Точность):   {precision:.4f} ({precision*100:.1f}%)
    • Recall (Полнота):       {recall:.4f} ({recall*100:.1f}%)
    • mAP@50:                 {map50:.4f} ({map50*100:.1f}%)
    • mAP@50-95:              {map50_95:.4f} ({map50_95*100:.1f}%)
    ===========================================================
    """
    print(report)

    with open("metrics_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    generate_metrics_report(
        weights_path="runs/detect/cones_yolov8s/weights/best.pt",
        data_yaml="dataset/data.yaml"
    )
