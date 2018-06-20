# Hyperparameter Tuner Abstract Class
import abc
from multiprocessing import Pool
import datetime
now = datetime.datetime.now

class BaseTuner(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        return NotImplemented

    @abc.abstractmethod
    def tune(self):
        return NotImplemented
        
    @abc.abstractmethod
    def save(self):
        return NotImplemented
    
    @property
    def time_elapsed(self):
        try:
            return (now()-self.start).total_seconds()
        except AttributeError:
            self.start = now()
            return 0
    
    @property
    def pool(self):
        try:
            return self._pool
        except AttributeError:
            self._pool = Pool()