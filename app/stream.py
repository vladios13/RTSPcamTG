import os
import cv2
import threading
import time

from app import state

# RTSP read/open timeout. Lower = faster death detection, but too low false-trips on a
# slow-delivering camera. Tune here (calibration knob) — 5s is aggressive, 10s is safer.
RTSP_TIMEOUT_MS = 5000

# Force RTSP over TCP; timeout (µs) prevents VideoCapture() from blocking on a dead link.
# FFmpeg 5.x renamed the RTSP socket option stimeout->timeout; the old name is ignored (default 30s).
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;%d' % (RTSP_TIMEOUT_MS * 1000)

_stream_threads: list = []


def _make_cap(url):
    """Open VideoCapture with minimal buffer and 5s open/read timeouts.
    Timeouts must go through the constructor params: cap.set() after open does NOT
    reach the FFmpeg interrupt callback (that's why reads hung for the default 30s)."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS,
    ])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _reconnect_delay(attempt):
    """Exponential backoff: 5s, 10s, 20s, 40s, capped at 60s."""
    return min(5 * (2 ** attempt), 60)


def processStream(name, url):
    state.logger.debug('processStream thread started: ' + name)

    if state.args.debug is None:
        counter = 0
        err = 0
        reconnect_attempt = 0
        ever_connected = False
        cap = _make_cap(url)
        try:
            while True:
                if state.stopStreams:
                    state.logger.debug('Exiting thread name: ' + name)
                    break
                if not cap.isOpened():
                    cap.release()
                    delay = _reconnect_delay(reconnect_attempt)
                    reconnect_attempt += 1
                    if not ever_connected:
                        state.logger.warning(
                            'Stream %s: initial connection failed (attempt %d), retrying in %ds',
                            name, reconnect_attempt, delay,
                        )
                        state.increase_counter('stream_connect_failures')
                    else:
                        state.logger.warning('Stream %s: cap not opened, reconnecting in %ds...', name, delay)
                        state.increase_counter('stream_resets')
                    err = 0
                    counter = 0
                    time.sleep(delay)
                    cap = _make_cap(url)
                    continue
                read_start = time.monotonic()
                ret, frame = cap.read()
                if ret:
                    counter += 1
                    err = 0
                    reconnect_attempt = 0
                    ever_connected = True
                    state.framebuffer[name] = frame
                else:
                    err += 1
                    # A read that blocked ~the full timeout is a real stall: the interrupt
                    # callback aborted FFmpeg mid-RTP-packet, so the cap is desynced and keeps
                    # yielding "Too short data"/corrupt macroblocks. Reopening is the only fix —
                    # don't keep reading the broken handle.
                    stalled = (time.monotonic() - read_start) * 1000 >= RTSP_TIMEOUT_MS * 0.8
                    if stalled or err > 20:
                        reason = 'read timeout, desynced' if stalled else '%d consecutive errors' % err
                        state.logger.warning('Stream %s: %s, reconnecting...', name, reason)
                        state.increase_counter('stream_resets')
                        cap.release()
                        err = 0
                        counter = 0
                        # Don't create cap here: let not cap.isOpened() on next iteration
                        # handle backoff and creation in a single place
                    else:
                        if err % 10 == 1:
                            state.logger.debug('Stream %s: no frame (err=%d)', name, err)
                        time.sleep(0.5)
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
    state.framebuffer = {}
    _stream_threads = []
    startup_delay = state.config.get('stream_startup_delay_s', 2)
    streams = state.config['streams']
    for i, s in enumerate(streams):
        t = threading.Thread(
            target=processStream,
            name=s['label'],
            args=(s['label'], s['url'],),
            daemon=True,
        )
        t.start()
        _stream_threads.append(t)
        # Stagger thread starts to avoid simultaneous FFmpeg RTSP handshakes
        if i < len(streams) - 1:
            time.sleep(startup_delay)
