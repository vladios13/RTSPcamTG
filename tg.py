from pprint import pprint
import telegram
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import shared

bot = None


def initBot():
    global bot
    if 'tg_token' in shared.config:
        bot = telegram.Bot(token=shared.config['tg_token'])

initBot()


def echo(update: Update, context: CallbackContext):
    """Echo the user message."""
    global bot
    print('Recieved msg '+update.message.text+' from '+str(update.effective_user.id)+' in chat: '+str(update.effective_chat.id))

    if update.message.text == '/ustop':
        print('stopping detection')
        shared.stopDetection = True
        bot.send_message(chat_id=update.effective_chat.id, text="Stopping detection")

    if update.message.text == '/ustart':
        shared.stopDetection = False
        bot.send_message(chat_id=update.effective_chat.id, text="Continue detection")

def begin():
    updater = Updater(bot.token, use_context=True)

    dp = updater.dispatcher
    

    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.

    return updater

if __name__ == "__main__":
    updater = begin()
    updater.idle()
