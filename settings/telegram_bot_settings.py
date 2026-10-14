"""
Модуль настроек Telegram-бота для отправки уведомлений о ходе симуляции.
Значения токена и идентификатора чата считываются из переменных окружения.
"""

import os

token: str = os.environ.get("NMSIM_TELEGRAM_BOT_TOKEN", "")
user_id: str = os.environ.get("NMSIM_TELEGRAM_CHAT_ID", "")
chat_id: str = user_id

__all__ = [
    "token",
    "user_id",
    "chat_id",
]

