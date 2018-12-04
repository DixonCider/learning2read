"""
All the dirty job for b05502055
"""

import datetime
import random
from collections import defaultdict
import torch
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

import learning2read
from learning2read.utils import alod, draw, Index, IndexFold, list_diff, dict_to_code
from learning2read.io import PathMgr, DataMgr, save_pickle, load_pickle
from learning2read.proc import Procedure
from learning2read.preprocessing import RowFilter

now = datetime.datetime.now

R = random.Random()

PATH_LIN2 = {
    'data' : r"/tmp2/b05502055/data",
    'cache' : r"/tmp2/b05502055/data/cache",
    'doc' : r"/tmp2/b05502055/mltech",
}

PATH_LOCAL = {
    'data' : r"~/Documents/mltechFinal/src/data/raw"
}

PATH_LOCAL_PROCESSED = {
    'data' : r"~/Documents/mltechFinal/src/data/processed"
}

File = PathMgr(PATH_LIN2['cache'])
Data = DataMgr(PATH_LIN2['data'])
Doc = PathMgr(PATH_LIN2['doc'])

DataLocal = DataMgr(PATH_LOCAL['data'])

def preprocessing_for_vae(df_total):
    """
    Input : df_Total
    Output : x_train tensor for VAE
    """
    dfg_user = df_total.groupby('User-ID')
    gdf_user = dfg_user.agg({'Book-Rating':['count', 'min', 'max']})
    gdf_user.columns = ['count', 'min', 'max'] # cancel multilevel index
    gdf_user_gte10 = gdf_user.loc[gdf_user['count'] >= 1000, 'count'].sort_values(ascending=False)

    gdf_book = df_total.groupby('ISBN').agg({'User-ID':'count'})
    gdf_book.columns = ['count']
    gdf_book = gdf_book.loc[gdf_book['count'] >= 2, :] # cut
    # gdf_book=gdf_book.loc[gdf_book['count']>=300,:] # cut2
    gdf_book = gdf_book.sort_values('count', ascending=False)
    dim1 = len(gdf_book.index)
    dim2 = len(gdf_user_gte10.index)

    user_vector_id = defaultdict(lambda: -1)
    book_id = defaultdict(lambda: -1)
    i = 0
    for x in gdf_user_gte10.index:
        user_vector_id[x] = i
        i += 1
    i = 0
    for x in gdf_book.index:
        book_id[x] = i
        i += 1

    index_list = []
    for r in df_total.to_dict('record'):
        uid = user_vector_id[r['User-ID']]
        bid = book_id[r['ISBN']]
        if uid >= 0 and bid >= 0:
            index_list.append([bid, uid])

    index_tns = torch.LongTensor(index_list).t()
    value_tns = torch.ones(index_tns.size(1))
    train_tns = torch.sparse.FloatTensor(index_tns, value_tns, torch.Size([dim1, dim2]))
    train_tns = train_tns.to_dense()

    return train_tns
