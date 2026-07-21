import asyncio
import threading
import time
from typing import Optional

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile

from app import state
from app.utils import zero_division

STALE_FRAME_S = 30  # кадр старше — камера помечается в /status как проблемная

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
        return f'{d}д {h}ч {m}м'
    return f'{h}ч {m}м' if h else f'{m}м {s}с'


def _status_text():
    """Текст ответа на /status: аптайм, счётчики, свежесть кадров по камерам."""
    now = time.time()
    processed = state.get_counter('images_processed')
    avg = zero_division(state.get_counter('images_time'), processed)

    cameras = []
    for s in state.config.get('streams', []):
        ts = state.last_frame_ts.get(s['label'])
        if ts is None:
            cameras.append(f"  {s['label']} — нет кадров (!)")
            continue
        age = now - ts
        mark = ' (!)' if age > STALE_FRAME_S else ''
        cameras.append(f"  {s['label']} — кадр {int(age)} сек назад{mark}")

    detection = 'на паузе — /ustart для возобновления' if state.stopDetection else 'включена'

    return '\n'.join([
        '<b>RTSPcamTG</b>',
        f'Аптайм: {_fmt_duration(now - state.start_time)}',
        f'Детекция: {detection}',
        '',
        f"Тревог с запуска: {state.get_counter('alarms')}",
        f'Обработано кадров: {processed} (среднее {avg:.2f} с)',
        f"Пропущено: {state.get_counter('images_skipped')}",
        f"Реконнекты: {state.get_counter('stream_resets')} / "
        f"неудачных подключений: {state.get_counter('stream_connect_failures')}",
        '',
        'Камеры:',
        *(cameras or ['  нет настроенных камер']),
    ])


def initBot():
    global _bot, _dp
    if 'tg_token' not in state.config:
        state.logger.warning('notifier: tg_token not found in config, bot disabled')
        return
    _bot = Bot(token=state.config['tg_token'])
    _dp = Dispatcher()
    router = Router()
    _dp.include_router(router)

    @router.message(Command('ustop'))
    async def cmd_ustop(message: Message):
        state.stopDetection = True
        state.logger.info('Detection stopped via Telegram /ustop')
        await message.answer("Stopping detection")

    @router.message(Command('ustart'))
    async def cmd_ustart(message: Message):
        state.stopDetection = False
        state.logger.info('Detection resumed via Telegram /ustart')
        await message.answer("Continue detection")

    @router.message(Command('status'))
    async def cmd_status(message: Message):
        await message.answer(_status_text(), parse_mode=ParseMode.HTML)


async def _sender_worker():
    while True:
        task = await _queue.get()
        try:
            await _bot.send_photo(
                chat_id=task['chat_id'],
                photo=FSInputFile(task['photo_path']),
                caption=task['caption'],
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            state.logger.error('Telegram send_photo failed: %s', e)
        finally:
            _queue.task_done()


def _run_bot_loop():
    global _loop, _queue, _sender_task
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _queue = asyncio.Queue(maxsize=50)
    _sender_task = _loop.create_task(_sender_worker())
    state.logger.info('Aiogram event loop created, starting polling...')
    try:
        _loop.run_until_complete(_dp.start_polling(_bot, handle_signals=False))
    except Exception as e:
        state.logger.error('Aiogram polling crashed: %s', e, exc_info=True)
    finally:
        _sender_task.cancel()
        try:
            _loop.run_until_complete(_sender_task)
        except (asyncio.CancelledError, Exception):
            pass
        try:
            _loop.run_until_complete(_bot.session.close())
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


def send_alarm_photo(chat_id, photo_path: str, caption: str):
    """Синхронный мост: вызывается из detector-потока, отправляет фото через async bot."""
    if _bot is None:
        state.logger.warning('send_alarm_photo: _bot is None — tg_token настроен в config.json?')
        return
    if _loop is None:
        state.logger.warning('send_alarm_photo: _loop is None — notifier.begin() был вызван?')
        return
    if not _loop.is_running():
        state.logger.warning('send_alarm_photo: event loop не запущен — polling упал? Смотри ошибки выше')
        return

    async def _enqueue():
        try:
            _queue.put_nowait({'chat_id': chat_id, 'photo_path': photo_path, 'caption': caption})
        except asyncio.QueueFull:
            state.logger.warning('send_alarm_photo: queue full, dropping alert')

    asyncio.run_coroutine_threadsafe(_enqueue(), _loop)


initBot()
