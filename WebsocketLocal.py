class WSLocal:    
    def __init__(self):
        self.id = ''

    def get_id(self):
        return self.id
    
    def set_id(self, ws_id):
        self.id = ws_id
    
    def on_close(self):
        return self.id
    