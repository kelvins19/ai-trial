import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        elapsed_time = (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        return elapsed_time

    def get_elapsed_time(self):
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer has not been started or stopped.")
        
        elapsed_time = (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        self.reset()
        
        return elapsed_time
    
    def reset(self): 
        self.start_time = None
        self.end_time = None
