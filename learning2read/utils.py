from difflib import ndiff
from functools import reduce
from path import Path # package "path.py"
import pandas as pd
import numpy as np
import scipy
import pickle

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
def list_diff(alst,blst):
    return list(set(alst) - set(blst))

def dict_to_code(data):
    """
    for caching purpose
    diff dict -> diff code
    same code -> very likly same dict

    dict will be sorted at a shallow level...QQ
    """
    if type(data)==list:
        assert type(data[0])==dict
        return abs(hash(str( [sorted(d.items()) for d in data] )))
    return abs(hash(str(sorted(data.items()))))

def save_pickle(fpath,data):
    with open(fpath,'wb') as f:
        pickle.dump(data,f)
        
def load_pickle(fpath):
    with open(fpath,'rb') as f:
        data=pickle.load(f)
    return data

# 
def alod(df):
    """
    (DataFrame) as list of dict
    for loop generator
    """
    assert isinstance(df,pd.core.frame.DataFrame)
    for rpair in df.iterrows():
        d = dict(rpair[1])
        d['index'] = rpair[0]
        yield d

def check_array(array_like_object):
    return np.array(array_like_object)
try: # ugly
    import torch
    def check_tensor(array_like_object):
        return torch.from_numpy(array_like_object).float() # cause GPU does float faster
    def check_tensor_array(array_like_object):
        return check_tensor(check_array(array_like_object))
except:
    pass

def LCS(s1,s2): # length of LCS(s1,s2)
    diff=ndiff(s1,s2)
    result=0
    for part in diff:
        if part[0]==' ':
            result+=1
    return result
    
def better_match(key,s1,s2):
    """Determine whether [key] is [s1] or [s2]
    Parameters
    ----------
    key : str
        user input (like "btrain")
    s1 : str
        target1 (like "book_ratings_train.csv")
    s2 : str
        target2 (like "implicit_ratings.csv")

    Returns
    -------
    s1_better_match_than_s2 : bool
        ("book_ratings_train.csv" with LCS=6
        better than
        "implicit_ratings.csv" with LCS=5)
    """
    l1=LCS(key,s1)
    l2=LCS(key,s2)
    if l1==l2:
        return len(s1)<len(s2)
    return l1>l2
    

class DataLoader:
    def __init__(self,path=r"/Users/qtwu/Downloads/data",verbose=1):
        self.path=path
        self.verbose=verbose
        def File(name,**kwargs):
            return {
                'name' : name,
                'param' : kwargs
            }
        self.file_list=[
            File("book_ratings_test.csv"),
            File("book_ratings_train.csv"),
            File("books.csv"),
            File("implicit_ratings.csv"),
            File("submission.csv", header=None),
            File("users.csv"),
        ]
    def load(self,keyword,**kwargs):
        file_matched=reduce(
            lambda a,b:a if better_match(keyword,a['name'],b['name']) else b,
            self.file_list)
        fpath=Path(self.path).joinpath(file_matched['name'])
        if self.verbose:
            print(fpath)
        return pd.read_csv(fpath,**file_matched['param'])
    
