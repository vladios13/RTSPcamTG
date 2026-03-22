from __future__ import print_function

import os
import time
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass

import argparse
import json
import logging
import sys


def total_size(o, handlers={}, verbose=False):
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
        if verbose:
            print(s, type(o), repr(o), file=stderr)
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


framebuffer = {}
stats = {}
frame_stats = {}

stopStreams = False
stopDetection = False
stopProcess = False


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--debug', required=False, help='path to input image')
ap.add_argument('-od', '--outputdir', required=False, help='path to output folder', default='output')
ap.add_argument('-c', '--config', required=False, help='path to yolo config file', default='cfg/yolov4.cfg')
ap.add_argument('-w', '--weights', required=False, help='path to yolo pre-trained weights', default='cfg/yolov4.weights')
ap.add_argument('-cl', '--classes', required=False, help='path to text file containing class names', default='cfg/yolov4.txt')
ap.add_argument('-ic', '--invertcolor', required=False, help='invert RGB 2 BGR', default='false')
args = ap.parse_args()

if os.path.exists('config.json'):
    with open('config.json') as f:
        try:
            config = json.load(f)
        except Exception:
            config = {}
            config['streams'] = []
else:
    config = {}
    config['streams'] = []

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler('ultracam.log')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


def add_framestat(name, stat):
    data = []
    if name in frame_stats:
        data = frame_stats[name]
    data.append({'time': time.time(), 'stat': stat})
    frame_stats[name] = data


def get_size():
    return total_size(framebuffer) / 1024


def get_counter(name):
    return stats.get(name, 0)


def increase_counter(name, value=1):
    stats[name] = get_counter(name) + value
