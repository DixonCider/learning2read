# from .utils import DataLoader
import pandas as pd
import numpy as np
import scipy.stats
from collections import defaultdict
from .utils import alod

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
    @classmethod
    def run(cls,input_data,**kwargs):
        # WIP
        return {'output':output}
class UserBookTable:
    @classmethod
    def run(cls,input_data,**kwargs):
        # WIP
        return {'output':output}
        
class UserRatingSqueeze:
    @classmethod
    def run(cls,input_data,filter_num=1,statistics="mean",
    belongs_to='User-ID',objective='Book-Rating'):
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
        return {'output':pd.DataFrame([
            gen(iid,lst) for iid,lst in dict_of_list.items()])}

class BookRatingSqueeze:
    @classmethod
    def run(cls,**kwargs):
        return {'output':UserRatingSqueeze.run(belongs_to='ISBN',**kwargs)['output']}

class TotalDataFrame:
    @classmethod
    def run(cls,input_data,test_fill=-1,train_drop=None,**kwargs):
        assert type(input_data)==list
        output = pd.concat(input_data, axis=0)
        output = output.fillna(test_fill)
        return {'output':output}