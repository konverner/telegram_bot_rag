from telegram_bot_rag.api.telegram import start_bot
from telegram_bot_rag.db.database import create_tables

if __name__ == "__main__":
    create_tables()
    start_bot()
