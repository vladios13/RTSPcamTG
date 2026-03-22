import asyncio
import ctypes
import os
import platform
import signal
import threading

from app import state
from app import stream
from app import detector
from app import notifier
from app import server

_opencv_thread = None


def init_processnamehack():
    LIB = 'libcap.so.2'
    try:
        libcap = ctypes.CDLL(LIB)
    except OSError:
        print('Library {} not found. Unable to set thread name.'.format(LIB))
    else:
        def _name_hack(self):
            libcap.prctl(15, self.name.encode())  # PR_SET_NAME = 15
            threading.Thread._bootstrap_original(self)

        threading.Thread._bootstrap_original = threading.Thread._bootstrap
        threading.Thread._bootstrap = _name_hack


def _sigterm_handler(signum, frame):
    """SIGTERM (docker stop, systemd) — останавливаем Sanic для graceful shutdown."""
    state.logger.info('SIGTERM received, stopping web server...')
    server.app.stop()


if __name__ == '__main__':
    state.logger.info('Ultracam startup')

    if platform.system() == 'Linux':
        nice = os.nice(5)
        state.logger.info('nice level: {}'.format(nice))

    init_processnamehack()

    stream.loadStreams()

    detector.init_model()

    _opencv_thread = threading.Thread(target=detector.processFrame, name='opencv', daemon=True)
    _opencv_thread.start()

    notifier.begin()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    server.begin()  # блокирует; Sanic сам обрабатывает Ctrl+C (SIGINT) и возвращает

    # --- Graceful shutdown после остановки Sanic ---
    state.logger.info('Shutting down...')
    state.stopStreams = True
    state.stopProcess = True

    if notifier._loop is not None and notifier._loop.is_running() and notifier._dp is not None:
        asyncio.run_coroutine_threadsafe(notifier._dp.stop_polling(), notifier._loop)

    for t in stream._stream_threads:
        t.join(timeout=3)
    if _opencv_thread is not None:
        _opencv_thread.join(timeout=5)

    state.logger.info('Shutdown complete')
    print('Main end')
