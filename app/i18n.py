import json
from pathlib import Path

from app import state

DEFAULT_LANG = 'ru'
_LANG_DIR = Path(__file__).resolve().parent.parent / 'lang'
_strings = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in _LANG_DIR.glob('*.json')}


def t(key):
    """Строка UI на языке из config.json ('lang'); фолбэк — ru, затем сам ключ."""
    lang = state.config.get('lang') or DEFAULT_LANG
    return _strings.get(lang, {}).get(key) or _strings[DEFAULT_LANG].get(key) or key
