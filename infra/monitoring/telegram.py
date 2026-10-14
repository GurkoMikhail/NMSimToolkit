"""
Модуль интеграции с Telegram Bot API для отправки логов и уведомлений о статусе симуляции.
"""

from logging import StreamHandler
from pathlib import Path
from typing import Union

from telebot import TeleBot

from settings.telegram_bot_settings import token, user_id

ENV_FILE_PATH: Path = Path(__file__).resolve().parents[2] / ".env"


class TeleBotStream(TeleBot):
    """
    Потоковый адаптер Telegram-бота для перенаправления сообщений в чат.
    """

    user_id: Union[int, str]

    def __init__(self, bot_token: str = token, target_user_id: Union[int, str] = user_id) -> None:
        super().__init__(bot_token)
        self.user_id = target_user_id

    def write(self, message: str) -> None:
        """Отправка текстового сообщения в целевой чат Telegram."""
        self.send_message(self.user_id, message)


class TeleBotHandler(StreamHandler):  # type: ignore[type-arg]
    """
    Обработчик логов Python (logging.StreamHandler) для отправки записей через Telegram Bot.
    """

    def __init__(self, bot_token: str = token, target_user_id: Union[int, str] = user_id) -> None:
        super().__init__(TeleBotStream(bot_token, target_user_id))  # type: ignore[arg-type]


__all__ = [
    "ENV_FILE_PATH",
    "TeleBotStream",
    "TeleBotHandler",
]

