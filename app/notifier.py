import asyncio
import threading
from typing import Optional

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, FSInputFile

from app import state

_bot: Optional[Bot] = None
_dp: Optional[Dispatcher] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_tg_thread: Optional[threading.Thread] = None


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


def _run_bot_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    state.logger.info('Aiogram event loop created, starting polling...')
    try:
        _loop.run_until_complete(_dp.start_polling(_bot, handle_signals=False))
    except Exception as e:
        state.logger.error('Aiogram polling crashed: %s', e, exc_info=True)
    finally:
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

    future = asyncio.run_coroutine_threadsafe(
        _bot.send_photo(
            chat_id=chat_id,
            photo=FSInputFile(photo_path),
            caption=caption,
            parse_mode=ParseMode.HTML,
        ),
        _loop,
    )
    try:
        future.result(timeout=15)
    except Exception as e:
        state.logger.error('Telegram send_photo failed: %s', e)


initBot()
