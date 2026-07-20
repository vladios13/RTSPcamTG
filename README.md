# RTSPcamTG

Мониторинг IP-камер по RTSP с детекцией объектов (YOLOv8) и отправкой уведомлений в Telegram. Без облачных зависимостей.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-00FFFF?logo=onnx&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Telegram](https://img.shields.io/badge/Telegram-Bot-26A5E4?logo=telegram&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)

## Скриншоты

<!-- Главная страница (сетка камер) -->
<!-- Лента событий -->
<!-- Редактор конфига / polygon-зоны -->

## Быстрый старт

Веса модели (`cfg/yolov8n.onnx`, ~13 МБ) уже в репозитории — скачивать отдельно не нужно.

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/vladios13/RTSPcamTG.git
cd RTSPcamTG

# 2. Заполните конфиг
cp config.json.example config.json
# Откройте config.json и укажите tg_token, tg_chat и RTSP-потоки

# 3. Создайте директории для хранения детекций
mkdir -p output alarm

# 4. Запустите
docker compose build && docker compose up -d
```

Зоны детекции настраиваются через Web UI: `http://127.0.0.1:8000/config`

## Настройка

Отредактируйте `config.json` перед запуском:

```json
{
  "tg_token": "123456:ABC-DEF...",
  "tg_chat": -100123456789,
  "stream_startup_delay_s": 2,
  "streams": [
    {
      "label": "front_door",
      "url": "rtsp://user:pass@192.168.1.10/stream",
      "detect_in_polygon": [[100,200],[400,200],[400,500],[100,500]],
      "ignore": []
    }
  ]
}
```

`detect_in_polygon` — список точек (x, y). Редактор полигонов доступен в Web UI (`/config`).

## Web UI

Bootstrap 5.3.3, без jQuery, адаптивный. Шрифт Manrope, светлая тема, индиго акцент. Снапшоты камер обновляются каждые 2 секунды через onload-цепочку — без накопления очереди запросов к серверу.

| Путь (порт 8000) | Описание |
|---|---|
| `/` | Сетка live-снапшотов всех камер |
| `/config` | Настройка потоков, Telegram, polygon-зон |
| `/video/<tag>` | Live-просмотр одной камеры |
| `/movement` | Лента событий: последние 50 кадров из `alarm/`, фильтр по камере, пагинация |
| `/api/stats`, `/api/movement`, `/api/alarms` | JSON API |

## Как работает

- Система получает видео с камер по RTSP и анализирует каждый кадр через YOLOv8n (инференс через ONNX + `cv2.dnn`).
- Если в заданной зоне обнаружен человек (или другой отслеживаемый объект) — отправляется фото в Telegram с подписью: камера, время, что обнаружено.
- Одно и то же срабатывание не дублируется: повторные уведомления об одном объекте в той же точке подавляются на 5 минут.
- При обрыве соединения с камерой система автоматически переподключается с нарастающим интервалом: 5→10→20→40→60 сек.
- Данные сохраняются на диск: кропы объектов в `output/` и полные кадры тревог в `alarm/`.

Telegram-команды: `/ustop` — пауза детекции, `/ustart` — возобновить.

## Пересборка модели

Если нужно обновить или сменить веса — `scripts/export_yolov8.py` экспортирует `yolov8n.pt` в ONNX:

```bash
python -m venv /tmp/yolo-export && . /tmp/yolo-export/bin/activate
pip install ultralytics onnx onnxslim
python scripts/export_yolov8.py   # → cfg/yolov8n.onnx
```

## Зависимости

- Python 3.11
- OpenCV 4.9.0
- YOLOv8n (ONNX, ~13 МБ, в комплекте, скачивать отдельно не нужно)
- aiogram 3.7.0
- Sanic 24.12.0
- Shapely, Jinja2
- Docker / docker-compose (лимит CPU контейнера: 0.4)

---

Основано на [Nalorokk/ultracam](https://github.com/Nalorokk/ultracam), развивается независимо.
