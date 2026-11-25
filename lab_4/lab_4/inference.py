import os
import glob
from ultralytics import YOLO

def run_inference(weights_path, source_path):
    model = YOLO(weights_path)

    results = model.predict(
        source=source_path,
        conf=0.05,          # !!! порог уверенности (если модель уверена больше, чем на conf, рисуется рамка)
                            # !!! ~0.5 (понизить, если пропускает конусы, повысить, если выбирает лишние)
        iou=0.45,           # !!! порог перекрытия рамок
                            # !!! ~0.45 (понизить, если несколько рамок вокруг одного конуса, 
                            #            повысить, если несколько конусов в одной рамке)
        imgsz=640,          # стандартное разрешение картинок
        save=True,          # сохранить визуализацию с рамками
        save_txt=True,      # сохранить координаты в txt
        name="test_results",
        exist_ok=True
    )

    print(f"\nОбработка завершена. Результаты сохранены в: /test_results")
    for i, r in enumerate(results[:5]):
        boxes = r.boxes
        print(f"Файл {os.path.basename(r.path)}: найдено конусов — {len(boxes)}")

if __name__ == "__main__":
    WEIGHTS = "runs/detect/cones_yolov8s/weights/best.pt"
    TEST_SOURCE = "test_media/"
    run_inference(WEIGHTS, TEST_SOURCE)
