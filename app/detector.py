import cv2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import os.path
import datetime
import numpy as np
import time
import argparse

from app import state
from app import notifier

classes = None
allowed = [
    'car', 'bicycle', 'dog', 'motorbike', 'umbrella', 'boat', 'pottedplant',
    'fire hydrant', 'train', 'bus', 'bowl', 'cup', 'frisbee', 'bench',
    'tvmonitor', 'sports ball', 'bottle', 'bird', 'truck', 'banana',
    'surfboard', 'refrigerator', 'sheep', 'traffic light', 'aeroplane',
    'chair', 'diningtable', 'diningtable', 'suitcase', 'backpack',
    'vase',
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


def init_model():
    """Загружает YOLO-модель один раз при старте. Вызывается из main.py."""
    global net
    state.logger.info('Loading YOLO model: %s + %s', state.args.weights, state.args.config)
    net = cv2.dnn.readNet(state.args.weights, state.args.config)
    state.logger.info('YOLO model loaded successfully')
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        state.logger.info('YOLO: CUDA backend enabled')
    except Exception:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        state.logger.info('YOLO: CPU backend (CUDA unavailable)')


def get_output_layers(net):
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    # OpenCV <4.5 возвращает shape (n,1); OpenCV >=4.5 — плоский массив
    if unconnected.ndim == 2:
        return [layer_names[i[0] - 1] for i in unconnected]
    return [layer_names[i - 1] for i in unconnected]


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

    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
    object_lines = '\n'.join('  ' + line for line in alarm)

    caption = (
        f'<b>[ALARM]</b>  <b>{name}</b>\n'
        f'<i>{timestamp_str}</i>\n'
        f'\n'
        f'Detected ({len(alarm)}):\n'
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
    image = state.framebuffer[name]

    if state.stopDetection:
        return None

    if name2 in state.framebuffer:
        commutative_image_diff = get_image_difference(state.framebuffer[name], state.framebuffer[name2])
        state.increase_counter('total_diff', commutative_image_diff)
        state.increase_counter('total_processed')
        state.add_framestat(name, commutative_image_diff)

        # Порог 0.005 соответствует ~1.3/255 средней разнице пикселей
        if commutative_image_diff < 0.005:
            state.increase_counter('total_skip_diff', commutative_image_diff)
            state.logger.debug('[%s] skipping frame: diff=%.4f (below threshold)', name, commutative_image_diff)
            return None

    state.framebuffer[name2] = image

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    alarm = []
    polygon = None

    if 'detect_in_polygon' in stream:
        polygon = Polygon(stream['detect_in_polygon'])

    orgImage = image.copy()
    for i in indices:
        idx = i[0] if hasattr(i, '__len__') else i
        box = boxes[idx]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_prediction(image, class_ids[idx], confidences[idx], round(x), round(y), round(x + w), round(y + h))

        point_loc = round(x + w / 2), round(y + h / 2)
        point = Point(*point_loc)

        alarm_object_name = str(classes[class_ids[idx]])
        if alarm_object_name not in allowed and checkAlarm(alarm_object_name, point_loc, confidences[idx]):
            in_zone = polygon is None or polygon.contains(point)
            if in_zone:
                # Объект внутри зоны — алерт
                cv2.circle(image, point_loc, 5, (0, 0, 255), -1)
                alarm.append('{}: {:.2%}'.format(alarm_object_name, confidences[idx]))
                save_bounded_image(orgImage, class_ids[idx], confidences[idx], round(x), round(y), round(x + w), round(y + h))
            else:
                # Объект вне зоны — рисуем зелёный кружок, не уведомляем
                cv2.circle(image, point_loc, 5, (0, 255, 0), -1)
                state.logger.debug('Found %s outside detection zone, skipping', alarm_object_name)
        elif alarm_object_name in allowed:
            state.logger.debug('Ignored: %s (in allowed list)', alarm_object_name)

    if str2bool(state.args.invertcolor):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if len(alarm) > 0:
        perform_alarm(name, image, alarm)
    return image


def checkAlarm(name, point, confidence):
    """
    Антиспам алертов.

    Подавляет повторные уведомления об одном и том же объекте
    в одном месте в течение "WINDOW" секунд.

    Считается повтором если:
      - совпадает класс объекта (name)
      - объект находится в радиусе 30px от предыдущего (Manhattan distance)

    Возвращает True если алерт нужно отправить, False если подавить.
    """
    global notified
    now = time.time()
    WINDOW = 60 * 5  # Окно антиспама: 5 минут

    notified = [a for a in notified if now - a['time'] < WINDOW]

    for alarm in notified:
        if (alarm['name'] == name
                and abs(alarm['point'][0] - point[0]) + abs(alarm['point'][1] - point[1]) < 30):
            state.logger.debug('Antispam: suppressed %s conf=%.2f at (%d,%d), repeat within %ds window',
                               name, confidence, point[0], point[1], WINDOW)
            return False

    notified.append({'name': name, 'point': point, 'confidence': confidence, 'time': now})
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
