import time

from app import notifier, state


def test_initbot_without_token_drops_previous_bot(monkeypatch):
    monkeypatch.setattr(state, 'config', {'tg_token': '123456:TEST-token', 'tg_chat': 1})
    notifier.initBot()
    assert notifier._bot is not None

    monkeypatch.setattr(state, 'config', {'tg_chat': 1})
    notifier.initBot()
    assert notifier._bot is None
    assert notifier._dp is None


def test_initbot_empty_or_invalid_token_disables_bot(monkeypatch):
    for token in ('', 'garbage'):
        monkeypatch.setattr(state, 'config', {'tg_token': token, 'tg_chat': 1})
        notifier.initBot()
        assert notifier._bot is None
        assert notifier._dp is None


def test_parse_duration():
    assert notifier.parse_duration('90s') == 90
    assert notifier.parse_duration('30m') == 1800
    assert notifier.parse_duration('24h') == 86400
    assert notifier.parse_duration('1d') == 86400
    for bad in ('25h', '2d', '0m', '30', 'abc', '1h30m', '-5m', ''):
        assert notifier.parse_duration(bad) is None, bad


def test_parse_duration_rejects_non_ascii_digits():
    assert notifier.parse_duration('٣m') is None
    assert notifier.parse_duration('０5m') is None


def test_status_shows_timed_pause(monkeypatch):
    monkeypatch.setattr(state, 'stopDetection', True)
    monkeypatch.setattr(state, 'detection_paused_until', time.time() + 1800)
    text = notifier._status_text()
    assert notifier.t('bot.detection_paused_until').split('{')[0] in text
