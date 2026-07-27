import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import os.path
import datetime
import html
import numpy as np
import time
import argparse

from app import state
from app.i18n import t
from app import notifier

classes = None
# Имена классов — в нейминге ultralytics COCO (см. cfg/yolov8.txt).
# Отличия от darknet-имён YOLOv4: tvmonitor→tv, aeroplane→airplane,
# motorbike→motorcycle, pottedplant→potted plant, diningtable→dining table.
ignored_classes = [
    'tv', 'sports ball', 'bottle', 'bird', 'truck', 'bicycle', 'banana', 'surfboard',
    'refrigerator', 'sheep', 'traffic light', 'airplane', 'motorcycle', 'umbrella', 'chair',
    'boat', 'potted plant', 'fire hydrant', 'train', 'bus', 'bowl', 'cup', 'frisbee', 'bench',
    'dining table', 'suitcase', 'backpack', 'vase',
    'elephant', 'bear', 'teddy bear', 'zebra', 'giraffe', 'tennis racket', 'kite',
]

with open(state.args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

notified = []
net = None   # инициализируется через init_model() при старте


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _get_cuda_device_count():
    """Возвращает количество доступных CUDA-устройств, 0 если CUDA недоступна.
    
    """
    if not hasattr(cv2, 'cuda'):
        return 0
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        return count if count > 0 else 0
    except cv2.error as e:
        state.logger.debug('CUDA probe failed: %s', e)
        return 0


def init_model():
    """Загружает YOLOv8 ONNX-модель один раз при старте. Вызывается из main.py."""
    global net
    state.logger.info('Loading YOLOv8 ONNX model: %s', state.args.weights)
    net = cv2.dnn.readNetFromONNX(state.args.weights)
    state.logger.info('YOLOv8 ONNX model loaded successfully')

    cuda_devices = _get_cuda_device_count()
    if cuda_devices > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        state.logger.info('YOLO: CUDA backend enabled (%d device(s))', cuda_devices)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        state.logger.info('YOLO: CPU backend (CUDA unavailable)')


INPUT_SIZE = 416  # размер входа сети (совпадает с imgsz при экспорте ONNX)


def letterbox(img, new_shape=INPUT_SIZE, color=(114, 114, 114)):
    """Ресайз с сохранением пропорций и паддингом до квадрата new_shape×new_shape.

    Повторяет препроцессинг ultralytics (LetterBox), чтобы не искажать
    16:9-кадры при сжатии до квадрата. Возвращает (padded, r, (pad_left, pad_top)),
    где r — коэффициент масштаба, pad_* — смещения для обратного пересчёта боксов.
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = round(w * r), round(h * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = new_shape - new_w, new_shape - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def save_bounded_image(image, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    dirname = os.path.join(state.args.outputdir, label, datetime.datetime.now().strftime('%Y-%m-%d'))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = (
        label + '_' +
        datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f') +
        '_conf' + '{:.2f}'.format(confidence) + '.jpg'
    )
    state.logger.debug('Saving bounding box: %s', filename)
    roi = image[y:y_plus_h, x:x_plus_w]
    if roi.any():
        if not str2bool(state.args.invertcolor):
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(dirname, filename), roi)


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 3)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)


def perform_alarm(name, image, alarm):
    MSK = datetime.timezone(datetime.timedelta(hours=3))
    now = datetime.datetime.now(MSK)
    directory = os.path.join('alarm', name, now.strftime('%Y-%m-%d'))
    path = os.path.join(directory, now.strftime('%Y-%m-%d_%H_%M_%S_%f') + '.jpg')

    if not os.path.exists(directory):
        os.makedirs(directory)

    cv2.imwrite(path, image)
    state.logger.warning('[ALARM] cam=%s objects=%d path=%s', name, len(alarm), path)
    state.increase_counter('alarms')

    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
    object_lines = '\n'.join('  ' + line for line in alarm)

    caption = (
        f'<b>[ALARM]</b>  <b>{html.escape(name)}</b>\n'
        f'<i>{timestamp_str}</i>\n'
        f'\n'
        f"{t('bot.detected')} ({len(alarm)}):\n"
        f'{object_lines}'
    )

    # Telegram ограничивает caption фото до 1024 символов
    if len(caption) > 1024:
        caption = caption[:1021] + '...'

    if 'tg_chat' in state.config:
        notifier.send_alarm_photo(
            chat_id=state.config['tg_chat'],
            photo_path=path,
            caption=caption,
        )


def get_image_difference(image_1, image_2):
    """
    Нормализованная мера попиксельного различия кадров [0..1].
    0 = одинаковые кадры, больше = сильнее изменение.
    """
    if image_1.shape != image_2.shape:
        return 1.0
    diff = cv2.absdiff(image_1, image_2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if diff.ndim == 3 else diff
    return float(np.mean(gray)) / 255.0


def detect(stream):
    if net is None:
        state.logger.warning('detect() called before init_model()')
        return None

    name = stream['label']
    name2 = name + '_processed'

    frame = state.framebuffer.get(name)
    if frame is None:
        return None
    image = frame.copy()

    if state.stopDetection:
        return None

    prev_frame = state.framebuffer.get(name2)
    if prev_frame is not None:
        commutative_image_diff = get_image_difference(frame, prev_frame)
        state.increase_counter('total_diff', commutative_image_diff)
        state.increase_counter('total_processed')
        state.add_framestat(name, commutative_image_diff)

        # Порог 0.005 соответствует ~1.3/255 средней разнице пикселей
        if commutative_image_diff < 0.005:
            state.increase_counter('total_skip_diff', commutative_image_diff)
            state.logger.debug('[%s] skipping frame: diff=%.4f (below threshold)', name, commutative_image_diff)
            return None

    state.framebuffer[name2] = image

    # Letterbox-препроцессинг: паддинг с сохранением пропорций до INPUT_SIZE.
    padded, ratio, (pad_left, pad_top) = letterbox(image, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                 (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # Выход YOLOv8: (1, 84, N) — 4 коорд бокса (cx,cy,w,h) + 80 class scores.
    # Без objectness (в отличие от YOLOv4). Транспонируем в (N, 84).
    out = np.squeeze(net.forward()).T

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for row in out:
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            # Обратный letterbox-пересчёт в пиксели исходного кадра.
            x = (cx - w / 2 - pad_left) / ratio
            y = (cy - h / 2 - pad_top) / ratio
            class_ids.append(int(class_id))
            confidences.append(float(confidence))
            boxes.append([x, y, w / ratio, h / ratio])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    alarm = []
    polygon = None

    if stream.get('detect_in_polygon'):
        polygon = Polygon(stream['detect_in_polygon'])

    orgImage = frame
    for idx in indices:
        box = boxes[idx]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_prediction(image, class_ids[idx], confidences[idx], round(x), round(y), round(x + w), round(y + h))

        point_loc = round(x + w / 2), round(y + h / 2)
        point = Point(*point_loc)

        alarm_object_name = str(classes[class_ids[idx]])
        if alarm_object_name not in ignored_classes and checkAlarm(name, alarm_object_name, point_loc):
            in_zone = polygon is None or polygon.contains(point)
            if in_zone:
                # Объект внутри зоны — алерт
                cv2.circle(image, point_loc, 5, (0, 0, 255), -1)
                alarm.append(f'{alarm_object_name}: {confidences[idx]:.2%}')
                save_bounded_image(orgImage, class_ids[idx], confidences[idx], round(x), round(y), round(x + w), round(y + h))
            else:
                # Объект вне зоны — рисуем зелёный кружок, не уведомляем
                cv2.circle(image, point_loc, 5, (0, 255, 0), -1)
                state.logger.debug('Found %s outside detection zone, skipping', alarm_object_name)
        elif alarm_object_name in ignored_classes:
            state.logger.debug('Ignored: %s (in ignored_classes)', alarm_object_name)

    if str2bool(state.args.invertcolor):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(alarm) > 0:
        perform_alarm(name, image, alarm)
    return image


def checkAlarm(cam, name, point):
    """
    Антиспам алертов.

    Подавляет повторные уведомления об одном и том же объекте
    в одном месте в течение "WINDOW" секунд.
    Считается повтором если:
      - совпадает камера (cam)
      - совпадает класс объекта (name)
      - объект находится в радиусе 30px от предыдущего (Manhattan distance)

    Возвращает True если алерт нужно отправить, False если подавить.
    """
    global notified
    now = time.time()
    WINDOW = 60 * 5  # Окно антиспама: 5 минут

    notified = [a for a in notified if now - a['time'] < WINDOW]

    for alarm in notified:
        if (alarm['cam'] == cam
                and alarm['name'] == name
                and abs(alarm['point'][0] - point[0]) + abs(alarm['point'][1] - point[1]) < 30):
            state.logger.debug('Antispam: suppressed %s cam=%s at (%d,%d), repeat within %ds window',
                               name, cam, point[0], point[1], WINDOW)
            return False

    notified.append({'cam': cam, 'name': name, 'point': point, 'time': now})
    return True


def processFrame():
    state.logger.info('processFrame started, monitoring %d stream(s)', len(state.config['streams']))
    while not state.stopProcess:
        for stream_cfg in state.config['streams']:
            if stream_cfg['label'] in state.framebuffer:
                begin = time.time()

                framed = detect(stream_cfg)
                if framed is not None:
                    state.framebuffer[stream_cfg['label'] + '_framed'] = framed
                    took = time.time() - begin
                    state.increase_counter('images_processed')
                    state.increase_counter('images_time', took)
                    state.logger.debug('[%s] detection took: %.3fs', stream_cfg['label'], took)
                else:
                    took = time.time() - begin
                    state.increase_counter('images_skipped')
                    state.increase_counter('skipped_time', took)

        time.sleep(0.5)
    state.logger.info('processFrame exiting')
