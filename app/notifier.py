import asyncio
import re
import threading
import time
from datetime import datetime
from typing import Optional

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import Message, FSInputFile
from aiogram.utils.token import TokenValidationError

from app import state
from app.i18n import t
from app.utils import zero_division

_bot: Optional[Bot] = None
_dp: Optional[Dispatcher] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_tg_thread: Optional[threading.Thread] = None
_queue: Optional[asyncio.Queue] = None
_sender_task: Optional[asyncio.Task] = None


def _fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}{t('bot.unit_d')} {h}{t('bot.unit_h')} {m}{t('bot.unit_m')}"
    if h:
        return f"{h}{t('bot.unit_h')} {m}{t('bot.unit_m')}"
    return f"{m}{t('bot.unit_m')} {s}{t('bot.unit_s')}"


_UNITS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
MAX_PAUSE_S = 24 * 3600


def parse_duration(arg):
    """Аргумент /ustop → секунды: '30m' → 1800. Только N[smhd], 0 < N ≤ 24h, иначе None."""
    m = re.fullmatch(r'(\d+)([smhd])', arg.strip().lower())
    if not m:
        return None
    seconds = int(m[1]) * _UNITS[m[2]]
    return seconds if 0 < seconds <= MAX_PAUSE_S else None


def _pause_until(left):
    """Аргументы для строк «до HH:MM, ещё …»: left — секунд до конца паузы."""
    return {
        'until': datetime.fromtimestamp(state.detection_paused_until).strftime('%H:%M'),
        'left': _fmt_duration(max(0, left)),
    }


def _status_text():
    """Текст ответа на /status: аптайм, режим детекции, счётчики."""
    now = time.time()
    processed = state.get_counter('images_processed')
    avg = zero_division(state.get_counter('images_time'), processed)

    if not state.stopDetection:
        detection = t('bot.detection_on')
    elif state.detection_paused_until is None:
        detection = t('bot.detection_paused')
    else:
        detection = t('bot.detection_paused_until').format(**_pause_until(state.detection_paused_until - now))

    return '\n'.join([
        '<b>RTSPcamTG</b>',
        f"{t('bot.uptime')}: {_fmt_duration(now - state.start_time)}",
        f"{t('bot.detection')}: {detection}",
        '',
        f"{t('bot.alarms')}: {state.get_counter('alarms')}",
        f"{t('bot.processed')}: {processed} ({t('bot.avg')} {avg:.2f} {t('bot.unit_s')})",
        f"{t('bot.skipped')}: {state.get_counter('images_skipped')}",
        f"{t('bot.reconnects')}: {state.get_counter('stream_resets')} / "
        f"{t('bot.connect_failures')}: {state.get_counter('stream_connect_failures')}",
    ])


_ADMIN_STATUSES = {'creator', 'administrator'}


async def _is_allowed(message: Message) -> bool:
    """Доступ к командам: tg_admins, владелец в ЛС, админы супергруппы tg_chat. Иначе — отказ."""
    user = message.from_user
    if user is None:  # посты канала
        return False

    tg_chat = state.config.get('tg_chat')
    if user.id in (state.config.get('tg_admins') or []):
        return True

    if tg_chat and message.chat.id == tg_chat:
        if message.chat.type == 'private':
            return True
        try:
            member = await _bot.get_chat_member(tg_chat, user.id)
        except Exception as e:
            state.logger.warning('get_chat_member(%s, %s) failed: %s', tg_chat, user.id, e)
        else:
            if member.status in _ADMIN_STATUSES:
                return True

    log = state.logger.warning if (message.text or '').startswith('/') else state.logger.debug
    log('Отклонено сообщение от user_id=%s в chat_id=%s: %r', user.id, message.chat.id, message.text)
    return False


def initBot():
    global _bot, _dp
    _bot = _dp = None
    if not state.config.get('tg_token'):
        state.logger.warning('notifier: tg_token not found in config, bot disabled')
        return
    if not state.config.get('tg_chat') and not state.config.get('tg_admins'):
        state.logger.warning('notifier: нет tg_chat и tg_admins — команды бота недоступны никому')
    try:
        _bot = Bot(token=state.config['tg_token'])
    except TokenValidationError:
        state.logger.error('notifier: tg_token is invalid, bot disabled')
        return
    _dp = Dispatcher()
    router = Router()
    router.message.filter(_is_allowed)
    _dp.include_router(router)

    @router.message(Command('ustop'))
    async def cmd_ustop(message: Message, command: CommandObject):
        if not command.args:
            state.detection_paused_until = None
            state.stopDetection = True
            state.logger.info('Detection stopped via Telegram /ustop')
            await message.answer(t('bot.detection_stopped'))
            return
        seconds = parse_duration(command.args)
        if seconds is None:
            await message.answer(t('bot.ustop_usage'))
            return
        state.detection_paused_until = time.time() + seconds
        state.stopDetection = True
        state.logger.info('Detection paused via Telegram /ustop %s', command.args)
        await message.answer(t('bot.detection_stopped_until').format(**_pause_until(seconds)))

    @router.message(Command('ustart'))
    async def cmd_ustart(message: Message):
        state.stopDetection = False
        state.detection_paused_until = None
        state.logger.info('Detection resumed via Telegram /ustart')
        await message.answer(t('bot.detection_resumed'))

    @router.message(Command('status'))
    async def cmd_status(message: Message):
        await message.answer(_status_text(), parse_mode=ParseMode.HTML)


async def _sender_worker():
    while True:
        task = await _queue.get()
        try:
            if 'photo_path' in task:
                await _bot.send_photo(
                    chat_id=task['chat_id'],
                    photo=FSInputFile(task['photo_path']),
                    caption=task['caption'],
                    parse_mode=ParseMode.HTML,
                )
            else:
                await _bot.send_message(chat_id=task['chat_id'], text=task['text'])
        except Exception as e:
            state.logger.error('Telegram send failed: %s', e)
        finally:
            _queue.task_done()


def _run_bot_loop():
    global _loop, _queue, _sender_task
    bot, dp = _bot, _dp
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _queue = asyncio.Queue(maxsize=50)
    _sender_task = _loop.create_task(_sender_worker())
    state.logger.info('Aiogram event loop created, starting polling...')
    try:
        _loop.run_until_complete(dp.start_polling(bot, handle_signals=False))
    except Exception as e:
        state.logger.error('Aiogram polling crashed: %s', e, exc_info=True)
    finally:
        _sender_task.cancel()
        try:
            _loop.run_until_complete(_sender_task)
        except (asyncio.CancelledError, Exception):
            pass
        try:
            _loop.run_until_complete(bot.session.close())
        except Exception:
            pass
        _loop.close()
    state.logger.info('Aiogram bot loop exited')


def stop():
    """Останавливает текущий polling если он запущен."""
    global _loop, _tg_thread
    if _loop is not None and _loop.is_running() and _dp is not None:
        try:
            asyncio.run_coroutine_threadsafe(_dp.stop_polling(), _loop)
            if _tg_thread is not None:
                _tg_thread.join(timeout=5)
        except Exception as e:
            state.logger.warning('notifier.stop(): %s', e)


def begin():
    global _tg_thread
    if _bot is None:
        state.logger.warning('notifier.begin(): bot not initialized, skipping')
        return
    _tg_thread = threading.Thread(target=_run_bot_loop, name='aiogram', daemon=True)
    _tg_thread.start()
    state.logger.info('Aiogram polling thread started')


def _submit(task):
    """Синхронный мост: кладёт задачу в очередь отправки из чужого потока (detector)."""
    if _bot is None:
        state.logger.warning('notifier: _bot is None — tg_token настроен в config.json?')
        return
    if _loop is None:
        state.logger.warning('notifier: _loop is None — notifier.begin() был вызван?')
        return
    if not _loop.is_running():
        state.logger.warning('notifier: event loop не запущен — polling упал? Смотри ошибки выше')
        return

    async def _enqueue():
        try:
            _queue.put_nowait(task)
        except asyncio.QueueFull:
            state.logger.warning('notifier: queue full, dropping message')

    asyncio.run_coroutine_threadsafe(_enqueue(), _loop)


def send_alarm_photo(chat_id, photo_path: str, caption: str):
    _submit({'chat_id': chat_id, 'photo_path': photo_path, 'caption': caption})


def send_text(chat_id, text: str):
    _submit({'chat_id': chat_id, 'text': text})
