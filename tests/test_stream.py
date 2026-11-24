import time

from app import state, stream


def test_loadstreams_restarts_without_duplicates(monkeypatch):
    """Повторный loadStreams() джойнит старые нити и не оставляет дублей на камеру."""
    monkeypatch.setattr(state, 'config', {
        'streams': [{'label': 'cam1', 'url': 'rtsp://x'}],
        'stream_startup_delay_s': 0,
    })
    started = []

    def fake_process(name, url):
        started.append(name)
        while not state.stopStreams:
            time.sleep(0.02)

    monkeypatch.setattr(stream, 'processStream', fake_process)

    stream.loadStreams()
    first = list(stream._stream_threads)
    stream.loadStreams()  # как двойное /config.save без отмены таймера

    assert started.count('cam1') == 2
    assert len(stream._stream_threads) == 1
    assert not first[0].is_alive()
