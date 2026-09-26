from logging import StreamHandler
from typing import Any, Optional, Union

from telebot import TeleBot

from settings.telegram_bot_settings import token, user_id


class TeleBotStream(TeleBot):
    """
    Потоковый адаптер Telegram-бота для перенаправления сообщений в чат.
    """

    def __init__(self, token: str = token, user_id: Union[int, str] = user_id) -> None:
        super().__init__(token)
        self.user_id = user_id

    def write(self, message: str) -> None:
        """Отправка текстового сообщения в целевой чат Telegram."""
        self.send_message(self.user_id, message)


class TeleBotHandler(StreamHandler):
    """
    Обработчик логов Python (logging.StreamHandler) для отправки записей через Telegram Bot.
    """

    def __init__(self, token: str = token, user_id: Union[int, str] = user_id) -> None:
        super().__init__(TeleBotStream(token, user_id))  # type: ignore


__all__ = [
    'TeleBotStream',
    'TeleBotHandler',
]
