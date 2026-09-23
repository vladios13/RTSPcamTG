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
