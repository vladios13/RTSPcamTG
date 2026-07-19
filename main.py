import os
import platform
import threading

from app import state
from app import stream
from app import detector
from app import notifier
from app import server


if __name__ == '__main__':
    state.logger.info('RTSPcamTG startup')

    if platform.system() == 'Linux':
        nice = os.nice(5)
        state.logger.info('nice level: {}'.format(nice))

    # Сигналы SIGINT/SIGTERM ловит сам Sanic (register_sys_signals=True по умолчанию):
    # при сигнале app.run() штатно завершается, дальше отрабатывает cleanup ниже.
    stream.loadStreams()

    detector.init_model()

    _opencv_thread = threading.Thread(target=detector.processFrame, name='opencv', daemon=True)
    _opencv_thread.start()

    notifier.begin()

    server.begin()  # блокирует; Sanic сам обрабатывает Ctrl+C (SIGINT)

    # --- Graceful shutdown после остановки Sanic ---
    state.logger.info('Shutting down...')
    state.stopStreams = True
    state.stopProcess = True

    notifier.stop()

    for t in stream._stream_threads:
        t.join(timeout=3)
    _opencv_thread.join(timeout=5)

    state.logger.info('Shutdown complete')
    state.logger.info('Main end')
