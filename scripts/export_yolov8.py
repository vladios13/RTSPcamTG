#!/usr/bin/env python3
"""Экспорт YOLOv8n в ONNX для инференса через cv2.dnn.

Запускается ОДИН РАЗ локально/в CI в отдельном venv с ultralytics
(НЕ в прод-контейнере — torch в прод не нужен). Результат кладётся
в cfg/yolov8n.onnx и коммитится в git.

Использование:
    python -m venv /tmp/yolo-export && . /tmp/yolo-export/bin/activate
    pip install ultralytics onnx onnxslim
    python scripts/export_yolov8.py

Параметры экспорта:
    imgsz=416   — фиксированный вход, совпадает с blob в detector.detect()
    opset=12    — совместимость с cv2.dnn (OpenCV 4.5.x)
    simplify    — упрощение графа (уменьшает риск неподдерживаемых операторов)
"""
import os
import shutil

from ultralytics import YOLO

CFG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'cfg')


def main():
    model = YOLO('yolov8n.pt')  # веса скачаются автоматически при первом запуске
    exported = model.export(format='onnx', imgsz=416, opset=12, simplify=True)

    target = os.path.abspath(os.path.join(CFG_DIR, 'yolov8n.onnx'))
    if os.path.abspath(exported) != target:
        shutil.move(exported, target)
    print('ONNX сохранён в', target)


if __name__ == '__main__':
    main()
