# camTGrtsp

Мониторинг IP-камер по RTSP с детекцией объектов (YOLOv4) и отправкой уведомлений в Telegram. Без облачных зависимостей.

## Скриншоты

<!-- Главная страница (сетка камер) -->
<!-- Лента событий -->
<!-- Редактор конфига / polygon-зоны -->

## Быстрый старт

```bash
# 1. Скачайте веса модели (~250 MB)
wget -q https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P cfg/

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
  "tg_chat": "-100123456789",
  "stream_startup_delay_s": 2,
  "streams": [
    {
      "label": "front_door",
      "url": "rtsp://user:pass@192.168.1.10/stream",
      "detect_in_polygon": [[100,200],[400,200],[400,500],[100,500]]
    }
  ]
}
```

`detect_in_polygon` — список точек (x, y). Редактор полигонов доступен в Web UI (`/config`).

## Web UI (порт 8000, только localhost)

Bootstrap 5.3.3, без jQuery, адаптивный. Шрифт Manrope, светлая тема, индиго акцент. Снапшоты камер обновляются каждые 2 секунды через onload-цепочку — без накопления очереди запросов к серверу.

| Путь | Описание |
|---|---|
| `/` | Сетка live-снапшотов всех камер |
| `/config` | Настройка потоков, Telegram, polygon-зон |
| `/video/<tag>` | Live-просмотр одной камеры |
| `/movement` | Лента событий: последние 50 кадров из `alarm/`, фильтр по камере, пагинация |
| `/api/stats`, `/api/movement`, `/api/alarms` | JSON API |

## Как работает

- Система получает видео с камер по RTSP и анализирует каждый кадр через YOLOv4.
- Если в заданной зоне обнаружен человек (или другой отслеживаемый объект) — отправляется фото в Telegram с подписью: камера, время, что обнаружено.
- Одно и то же срабатывание не дублируется: повторные уведомления об одном объекте в той же точке подавляются на 5 минут.
- При обрыве соединения с камерой система автоматически переподключается с нарастающим интервалом: 5→10→20→40→60 сек.
- Данные сохраняются на диск: кропы объектов в `output/` и полные кадры тревог в `alarm/`.

Telegram-команды: `/ustop` — пауза детекции, `/ustart` — возобновить.

## Зависимости

- Python 3.8
- OpenCV 4.5.5 (DNN inference)
- YOLOv4 (~250 MB, скачиваются отдельно)
- aiogram 3.7.0
- Sanic 19.12.5
- Shapely, Jinja2
- Docker / docker-compose (лимит CPU контейнера: 0.85)

---

Forked from [Nalorokk/ultracam](https://github.com/Nalorokk/ultracam)
