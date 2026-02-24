from logging import StreamHandler
from typing import Any, Optional, Union

from telebot import TeleBot

from settings.telegram_bot_settings import token, user_id


class TeleBotStream(TeleBot):
    
    def __init__(self, token: str = token, user_id: Union[int, str] = user_id) -> None:
        super().__init__(token)
        self.user_id = user_id
    
    def write(self, messenge: str) -> None:
        self.send_message(self.user_id, messenge)    
        
        
class TeleBotHandler(StreamHandler):
    
    def __init__(self, token: str = token, user_id: Union[int, str] = user_id) -> None:
        super().__init__(TeleBotStream(token, user_id)) # type: ignore
    
    