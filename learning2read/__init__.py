from importlib import reload
from . import utils
from . import plot
from . import preprocessing
from . import unsupervised
from . import submission
from . import proc
from . import io
from . import dnn
def reload_all(): # for module developing
    from . import b04
    reload(utils)
    reload(plot)
    reload(preprocessing)
    reload(unsupervised)
    reload(submission)
    reload(b04)
    reload(proc)
    reload(io)
    reload(dnn)

def reload_all_b05():
    """
    reloading modules when developing.
    """
    from . import b05
    reload(utils)
    reload(plot)
    reload(preprocessing)
    reload(unsupervised)
    reload(submission)
    reload(b05)
    reload(proc)
    reload(io)