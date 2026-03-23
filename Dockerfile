# Используем официальный образ Python 3.8 на базе slim-дистрибутива
FROM python:3.8-slim

WORKDIR /app

# Минимальные системные зависимости:
# - ffmpeg: декодирование RTSP-потоков
# - libsm6, libxext6: требования OpenCV
# - wget: скачивание весов YOLO при сборке
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

# Устанавливаем Python-зависимости из requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
