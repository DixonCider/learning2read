from importlib import reload
from . import utils
from . import plot
from . import preprocessing
from . import unsupervised
from . import submission
from . import proc
from . import io
from . import dnn
from . import tune

def reload_all(): # for module developing
    reload(utils)
    reload(plot)
    reload(preprocessing)
    reload(unsupervised)
    reload(submission)
    reload(proc)
    reload(io)
    reload(dnn)
    reload(tune)

def reload_all_b04():
    reload_all()
    from . import b04
    reload(b04)


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
    reload(tune)