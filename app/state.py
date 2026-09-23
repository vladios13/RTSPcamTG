import os
import time
from sys import getsizeof
from itertools import chain
from collections import deque

import json
import logging
import sys


def total_size(o, handlers={}):
    """Returns the approximate memory footprint of an object and all of its contents."""
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)
    seen = set()
    default_size = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


framebuffer = {}
stats = {}
frame_stats = {}
start_time = time.time()

stopStreams = False
stopDetection = False
detection_paused_until = None
stopProcess = False


OUTPUT_DIR = 'output'
WEIGHTS_PATH = 'cfg/yolov8n.onnx'
CLASSES_PATH = 'cfg/yolov8.txt'

config = {'streams': []}
logger = logging.getLogger()


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def init():
    """Читает config.json и настраивает логирование. Вызывается из main.py первым делом."""
    global config
    if os.path.exists('config.json'):
        with open('config.json') as f:
            try:
                config = json.load(f)
            except Exception:
                config = {'streams': []}

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger.setLevel(getattr(logging, str(config.get('log_level', 'INFO')).upper(), logging.INFO))

    fileHandler = logging.FileHandler('ultracam.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    sys.excepthook = handle_exception


def add_framestat(name, stat):
    data = []
    if name in frame_stats:
        data = frame_stats[name]
    data.append({'time': time.time(), 'stat': stat})
    # Ограничиваем буфер последними 3 часами (~2 точки/сек × 10800 сек = 21600 записей)
    frame_stats[name] = data[-21600:]


def get_size():
    return total_size(framebuffer) / (1024 * 1024)


def get_counter(name):
    return stats.get(name, 0)


def increase_counter(name, value=1):
    stats[name] = get_counter(name) + value
