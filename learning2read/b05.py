# "runner" classes made by b04303128 :p

from learning2read.utils import alod,draw,Index,IndexFold,list_diff,dict_to_code
from learning2read.io import PathMgr,DataMgr,save_pickle,load_pickle
from learning2read.proc import Procedure
from learning2read.preprocessing import RowFilter
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import datetime
now = datetime.datetime.now
import random
R = random.Random()

PATH_LIN2={
    'data' : r"/tmp2/b05502055/data",
    'cache' : r"/tmp2/b05502055/data/cache",
    'doc' : r"/tmp2/b05502055/mltech",
}

File = PathMgr(PATH_LIN2['cache'])
Data = DataMgr(PATH_LIN2['data'])
Doc = PathMgr(PATH_LIN2['doc'])

