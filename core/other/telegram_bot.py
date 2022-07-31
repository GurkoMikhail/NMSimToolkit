from telebot import TeleBot


class TeleBotStream(TeleBot):
    
    def __init__(self, token, user_id):
        super().__init__(token)
        self.user_id = user_id
    
    def write(self, messenge):
        self.send_message(self.user_id, messenge)
    #     self._messege = messenge
        
    # def flush(self)
    
    
    