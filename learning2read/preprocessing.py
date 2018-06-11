# from .utils import DataLoader
import pandas as pd
import numpy as np
import scipy.stats
from collections import defaultdict
from .utils import alod, LCS

class CleanRawUser:
    """
        'class' : 'learning2read.preprocessing.CleanRawUser',
        'output' : 'user_info',
        'input_data' : ['df_total', 'raw_user'],
        'na_policy' : 'median',
    """
    @classmethod
    def run(cls,input_data,na_policy):
        assert type(input_data)==list
        assert len(input_data)==2
        df_total = input_data[0]
        raw_user = input_data[1]
        user_info = pd.DataFrame([ {'User-ID':x} for x,_ in df_total.groupby('User-ID').indices.items()])
        user_info = user_info.merge(raw_user,on='User-ID',how='left')
        user_info['Loc_is_usa'] = user_info['Location'].apply(lambda r:int(LCS('usa',str(r))==3))
        user_info['Age_isna'] = user_info['Age'].apply(lambda r:int(pd.isna(r)))
        user_info = user_info.drop('Location',1)
        user_info = user_info.fillna(eval("user_info.%s()"%na_policy))
        return {'output' : user_info}

class CleanRawBook:
    """
        'class' : 'learning2read.preprocessing.CleanRawBook',
        'output' : 'book_info',
        'input_data' : ['df_total', 'raw_book'],
        'na_policy' : 'median',
    """
    @classmethod
    def run(cls,input_data,na_policy):
        assert type(input_data)==list
        assert len(input_data)==2
        df_total = input_data[0]
        raw_book = input_data[1]
        book_info = pd.DataFrame([ {'ISBN':x} for x,_ in df_total.groupby('ISBN').indices.items()])
        book_info = book_info.merge(raw_book,on='ISBN',how='left')
        book_info['ISBN_is_usa'] = book_info['ISBN'].apply(lambda r:int((str(r)[0])=='0') )
        book_info['Year-Of-Publication'] = book_info['Year-Of-Publication'].apply(lambda r: np.nan if r<1000 or r>2018 else r)
        book_info['Year_isna'] = book_info['Year-Of-Publication'].apply(lambda r:int(pd.isna(r)) )
        book_info = book_info[['ISBN','ISBN_is_usa','Year_isna','Year-Of-Publication']]
        book_info = book_info.fillna(eval("book_info.%s()"%na_policy))
        return {'output' : book_info}



STATS_AVAILABLE=['num','quantile','mean','mode','std','skew','kurtosis']
def list_to_statistics(values,name,arg=None):
    assert len(values)>0
    if name=='quantile':
        assert 0<=arg and arg<=1
        return np.percentile(values, arg*100)
    elif name=='num':
        return len(values)
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

class UserPadding:
    """
    'class' : 'learning2read.preprocessing.UserPadding',
    'output' : 'user_rating',
    'input' : ['df_total', 'user_rating'],
    'isna_name' : 'User-ID_no_book',
    'na_policy' : 'median',
    """
    @classmethod
    def run(cls,input_data,isna_name,na_policy,belongs_to='User-ID'):
        assert type(input_data)==list
        assert len(input_data)==2
        df_total = input_data[0]
        df_padding = input_data[1]
        output = pd.DataFrame([ {belongs_to:x} for x,_ in df_total.groupby(belongs_to).indices.items()])
        df_padding[isna_name] = 0
        output = output.merge(df_padding,on=belongs_to,how='left')
        output[isna_name] = output[isna_name].apply(lambda r:int(pd.isna(r)))
        output = output.fillna(eval("output.%s()"%na_policy))
        return {'output' : output}
        
class BookPadding:
    @classmethod
    def run(cls,**kwargs):
        return UserPadding.run(belongs_to='ISBN',**kwargs)

class UserRatingSqueeze:
    @classmethod
    def run(cls,input_data,filter_num=1,statistics="mean",
    belongs_to='User-ID',objective='Book-Rating',na_policy="median"):
        assert filter_num>0
        if type(statistics)==str:
            statistics = [statistics]

        lts_arg_list = []
        lts_arg_list_short = []
        for name in statistics:
            if name=='quantile11': # ugly
                for q in range(11):
                    lts_arg_list.append({'name':'quantile', 'arg':round(0.1*q,1) })
            else:
                lts_arg_list.append({'name':name})
            if name=='num':
                lts_arg_list_short.append({'name':name})
        lts_arg_list.append({'name':'is_short'})
        lts_arg_list_short.append({'name':'is_short'})

        dict_of_list = defaultdict(lambda:[])
        for r in alod(input_data):
            if r[objective]>0: # only training data counts
                dict_of_list[r[belongs_to]].append(r[objective])

        def gen(iid,lst):
            r={belongs_to:iid}
            target_list = lts_arg_list
            is_short = False
            if len(lst)<filter_num:
                target_list = lts_arg_list_short
                is_short = True
                # return r # skip too short list
            for kwargs in target_list:
                name = "%s_%s"%(belongs_to,kwargs['name'])
                try:
                    if kwargs['arg']:
                        name += "_"+str(kwargs['arg'])
                except:
                    pass
                if kwargs['name']=='is_short':
                    r[name] = 1 if is_short else 0
                else:
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
        output = pd.concat(input_data, axis=0, ignore_index=True, sort=False)
        output = output.fillna(test_fill)
        return {'output':output}
