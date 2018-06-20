# Hyperparameter Tuner Abstract Class
import abc
# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
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
        except:
            self.start = now()
            return 0
    
    # multiprocessing suffer from "XXX can't pickle" ... OTL
    # @property
    # def pool(self):
    #     try:
    #         self._pool
    #     except AttributeError:
    #         self._pool = Pool()
    #     return self._pool

class EpochBasedTuner(BaseTuner):
    pass