import os
import cv2
import threading
import time

from app import state

# RTSP timeouts (calibration knobs). READ = per-frame stall detection: lower = faster death
# detection, but too low false-trips on a slow-delivering camera. OPEN is looser because a
# slow-handshaking camera/NVR needs more room to complete DESCRIBE/SETUP/PLAY before the first frame.
RTSP_TIMEOUT_MS = 5000
RTSP_OPEN_TIMEOUT_MS = 10000

# Force RTSP over TCP; timeout (µs) prevents VideoCapture() from blocking on a dead link.
# FFmpeg 5.x renamed the RTSP socket option stimeout->timeout; the old name is ignored (default 30s).
# discardcorrupt: drop partially-decoded frames (lost mid-frame slice) instead of returning them
# half-smeared — read() gives ret=False for those, so the reconnect/skip logic handles them cleanly.
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    f'rtsp_transport;tcp|timeout;{RTSP_TIMEOUT_MS * 1000}|fflags;discardcorrupt'
)

_stream_threads: list = []


def _make_cap(url):
    """Open VideoCapture with minimal buffer and RTSP open/read timeouts.
    Timeouts must go through the constructor params: cap.set() after open does NOT
    reach the FFmpeg interrupt callback (that's why reads hung for the default 30s)."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_TIMEOUT_MS,
    ])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _reconnect_delay(attempt):
    """Exponential backoff: 5s, 10s, 20s, 40s, capped at 60s."""
    return min(5 * (2 ** attempt), 60)


def processStream(name, url):
    state.logger.debug(f'processStream thread started: {name}')

    if state.args.debug is None:
        err = 0
        reconnect_attempt = 0
        ever_connected = False
        cap = _make_cap(url)
        try:
            while True:
                if state.stopStreams:
                    state.logger.debug(f'Exiting thread name: {name}')
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
                    time.sleep(delay)
                    cap = _make_cap(url)
                    continue
                read_start = time.monotonic()
                ret, frame = cap.read()
                if ret:
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
                        cap.release()
                        err = 0
                        # stream_resets is counted by the not-cap.isOpened() branch on the next
                        # iteration; release here and let it handle backoff + reopen + counting.
                    else:
                        if err % 10 == 1:
                            state.logger.debug('Stream %s: no frame (err=%d)', name, err)
                        time.sleep(0.5)
        finally:
            cap.release()
            state.logger.debug(f'Released VideoCapture: {name}')

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
