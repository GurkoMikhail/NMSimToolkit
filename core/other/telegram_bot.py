from telebot import TeleBot
from logging import StreamHandler
from settings.telegram_bot_settings import token, user_id


class TeleBotStream(TeleBot):
    
    def __init__(self, token=token, user_id=user_id):
        super().__init__(token)
        self.user_id = user_id
    
    def write(self, messenge):
        self.send_message(self.user_id, messenge)    
        
        
class TeleBotHandler(StreamHandler):
    
    def __init__(self, token=token, user_id=user_id):
        super().__init__(TeleBotStream(token, user_id))
    
    