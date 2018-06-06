# from .utils import DataLoader
import pandas as pd
import numpy as np
import scipy.stats
from collections import defaultdict
from .utils import alod

import random
def Index(n, total_n, seed):
    R = random.Random() # for thread safe
    R.seed(seed)
    if n<1:
        n = max(1,round(n * total_n))
    return R.sample(range(total_n),n)
def IndexFold(k_fold, total_n, seed):
    R = random.Random() # for thread safe
    R.seed(seed)
    idx_list = []
    for i in range(total_n):
        idx_list.append(i%k_fold)
    R.shuffle(idx_list)
    folds = [[] for _ in range(k_fold)]
    for i,idx in enumerate(idx_list):
        folds[idx].append(i)
    return folds

STATS_AVAILABLE=['quantile','mean','mode','std','skew','kurtosis']
def list_to_statistics(values,name,arg=None):
    assert len(values)>0
    if name=='quantile':
        assert 0<=arg and arg<=1
        return np.percentile(values, arg*100)
    elif name=='mean':
        return sum(values) / len(values)
    elif name=='mode':
        return max(set(values), key=values.count)
    elif name=='std':
        return scipy.stats.tstd(values) if len(values)>1 else 0
    elif name=='skew':
        return scipy.stats.skew(values)
    elif name=='kurtosis':
        return scipy.stats.kurtosis(values)
    raise Exception("no such statistics named '%s'"%name)

class RowFilter:
    """
        'class'  : 'learning2read.preprocessing.RowFilter',
        'output' : 'df_train',
        'input_data' : 'df_total_features',
        'func' : r"lambda df:df['Book-Rating']>0",
    """
    @classmethod
    def run(cls,input_data,func,**kwargs):
        if type(func)==str:
            func = eval(func)
        output = input_data.loc[func(input_data),:]
        return {'output':output}

class UserBookTable:
    """
        'class'  : 'learning2read.preprocessing.UserBookTable',
        'output' : 'df_total_features', # (X,y={-1,0,1,2,...,10})
        'input_data' : ['df_total', 'user_rating', 'book_rating', 'book_vector'],
        'na_policy' : None, # should fill it before training
    """
    @classmethod
    def run(cls,input_data,na_policy=None,drop_columns=['User-ID','ISBN']):
        assert len(input_data)>=2
        _na_policy = na_policy or 'median'
        output = input_data[0]
        for i in range(1,len(input_data)):
            df_right = input_data[i]
            is_user = 'User-ID' in df_right.columns
            is_book = 'ISBN' in df_right.columns
            assert is_user ^ is_book # exactly one
            output = output.merge(
                df_right,
                on='User-ID' if is_user else 'ISBN',
                how='left')
        output = output.drop(drop_columns, 1)
        if output.isnull().values.any() and not na_policy:
            print("[UserBookTable] WARNING: found na but no na_policy assigend. fill with %s"%_na_policy)
        output = output.fillna(eval("output.%s()"%_na_policy))
        return {'output':output}
"""
def rating_merge(rating,user,book): # only users.csv, books disposed
    df=rating
    df=df.merge(user,on='User-ID',how='left')
    df=df.merge(book,on='ISBN',how='left')
    df=df.drop(,1)
    df=df.fillna(df.median()) # fill with median
    return df
df_train=rating_merge(raw_train,df_user_rate,df_book)
df_train.sample(10)
"""
        
class UserRatingSqueeze:
    @classmethod
    def run(cls,input_data,filter_num=1,statistics="mean",
    belongs_to='User-ID',objective='Book-Rating',na_policy="median"):
        assert filter_num>0
        if type(statistics)==str:
            statistics = [statistics]

        lts_arg_list = []
        for name in statistics:
            if name=='quantile11': # ugly
                for q in np.linspace(0,1,11):
                    lts_arg_list.append({'name':'quantile', 'arg':round(q,2)})
            else:
                lts_arg_list.append({'name':name})

        dict_of_list = defaultdict(lambda:[])
        for r in alod(input_data):
            dict_of_list[r[belongs_to]].append(r[objective])

        def gen(iid,lst):
            r={belongs_to:iid}
            if len(lst)<filter_num:
                return r # skip too short list
            for kwargs in lts_arg_list:
                name = "%s_%s"%(belongs_to,kwargs['name'])
                try:
                    if kwargs['arg']:
                        name += "_"+str(kwargs['arg'])
                except:
                    pass
                r[name] = list_to_statistics(lst, **kwargs)
            return r

        output = pd.DataFrame([gen(iid,lst) for iid,lst in dict_of_list.items()])
        output = output.fillna(eval("output.%s()"%na_policy))
        return {'output' : output}

class BookRatingSqueeze:
    @classmethod
    def run(cls,**kwargs):
        return {'output':UserRatingSqueeze.run(belongs_to='ISBN',**kwargs)['output']}

class TotalDataFrame:
    @classmethod
    def run(cls,input_data,test_fill=-1,train_drop=None,**kwargs):
        assert type(input_data)==list
        output = pd.concat(input_data, axis=0, ignore_index=True)
        output = output.fillna(test_fill)
        return {'output':output}