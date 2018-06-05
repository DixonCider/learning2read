from difflib import ndiff
from functools import reduce
from path import Path # package "path.py"
import pandas as pd
import numpy as np
import scipy
import pickle
def save_pickle(fpath,data):
    with open(fpath,'wb') as f:
        pickle.dump(data,f)
def load_pickle(fpath):
    with open(fpath,'rb') as f:
        data=pickle.load(f)
    return data

# for loop generator
def as_lod(df):
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
        self.file_list=[
            "book_ratings_test.csv",
            "book_ratings_train.csv",
            "books.csv",
            "implicit_ratings.csv",
            "submission.csv",
            "users.csv",
        ]
    def load(self,keyword,**kwargs):
        filename_matched=reduce(
            lambda a,b:a if better_match(keyword,a,b) else b,
            self.file_list)
        fpath=Path(self.path).joinpath(filename_matched)
        if self.verbose:
            print(fpath)
        return pd.read_csv(fpath,**kwargs)
    
