import os
import cv2
import threading
import time

from app import state

# Force RTSP over TCP to avoid RTP packet reordering (bad cseq errors)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

_stream_threads: list = []


def processStream(name, url):
    state.logger.debug('processStream thread started: ' + name)

    if state.args.debug is None:
        counter = 0
        err = 0
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            while True:
                if state.stopStreams:
                    state.logger.debug('Exiting thread name: ' + name)
                    break
                if not cap.isOpened():
                    state.logger.warning('Stream %s: cap not opened, reconnecting...', name)
                    cap.release()
                    time.sleep(5)
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    err = 0
                    counter = 0
                    state.increase_counter('stream_resets')
                    continue
                ret, frame = cap.read()
                if ret:
                    counter += 1
                    err = 0
                    state.framebuffer[name] = frame
                else:
                    err += 1
                    if err % 10 == 1:
                        state.logger.debug('Stream %s: no frame (err=%d)', name, err)
                    time.sleep(0.5)
                    if err > 20:
                        state.logger.warning('Stream %s: %d consecutive errors, reconnecting...', name, err)
                        state.increase_counter('stream_resets')
                        cap.release()
                        err = 0
                        counter = 0
                        time.sleep(10)
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        finally:
            cap.release()
            state.logger.debug('Released VideoCapture: ' + name)

    else:
        with open(state.args.debug, mode='rb') as file:
            state.framebuffer[name] = file.read()
            state.logger.debug('Loaded image as stream output')


def loadStreams():
    global _stream_threads
    state.stopStreams = False
    _stream_threads = []
    for s in state.config['streams']:
        t = threading.Thread(
            target=processStream,
            name=s['label'],
            args=(s['label'], s['url'],),
            daemon=True,
        )
        t.start()
        _stream_threads.append(t)
