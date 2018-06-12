# "runner" classes made by b04303128 :p

from learning2read.unsupervised import Pow2AutoEncoder
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

PATH_LIN2={
    'data' : r"/tmp2/b06902021/ML/data",
    'cache' : r"/tmp2/b06902021/ML/file",
}

File = PathMgr(PATH_LIN2['cache'])
Data = DataMgr(PATH_LIN2['data'])

