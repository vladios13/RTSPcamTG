# Используем официальный образ Python 3.8 на базе slim-дистрибутива
FROM python:3.8-slim

WORKDIR /app

# Минимальные системные зависимости:
# - ffmpeg: декодирование RTSP-потоков
# - libsm6, libxext6: требования OpenCV
# - wget: скачивание весов YOLO при сборке
RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

# Скачиваем веса YOLOv4 (~260 MB)
RUN wget -q https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights \
        -P /app/cfg/

# Устанавливаем Python-зависимости из requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
