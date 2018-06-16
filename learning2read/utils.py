from difflib import ndiff
from path import Path # package "path.py"
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
import pickle
import random

import math
# norm = scipy.stats.norm
class RandomBox(random.Random):
    def __init__(self, seed=None):
        super(RandomBox,self).__init__(seed)
    def draw(self, lower, upper, lower_bound, upper_bound, confidence=0.68, log=False):
        if log:
            assert lower>0 and upper>0
            lower = math.log(lower)
            upper = math.log(upper)
        mu = (lower+upper)*0.5
        q = 1-(1-confidence)/2
        sigma = (upper-mu)/norm.ppf(q)
        x = self.gauss(mu,sigma)
        if log:
            x = math.exp(x)
        return max(min(x,upper_bound),lower_bound)
    def draw_log(self, lower, upper, lower_bound, upper_bound, confidence=0.68):
        return self.draw(lower, upper, lower_bound, upper_bound, confidence, log=True)
    def draw_int(self, lower, upper, lower_bound, upper_bound, confidence=0.68):
        r = self.draw(lower, upper, lower_bound, upper_bound, confidence, log=False)
        return max(lower_bound, min(upper_bound, int(round(r))))
    @classmethod
    def demo(cls, N=100, seed=1):
        R = cls(seed)
        lst = []
        for _ in range(N):
            lst.append(R.draw_int(60, 80, 0, 100, 0.68))
        print(lst)
        lst = []
        for _ in range(N):
            lst.append(R.draw_int(20, 90, 0, 100, 0.33))
        print(lst)

def draw(lower, upper, lower_bound, upper_bound, confidence=0.68, log=False, python_random=None):
    if not python_random:
        python_random = random.Random()
    if log:
        assert lower>0 and upper>0
        lower = math.log(lower)
        upper = math.log(upper)
    
    mu = (lower+upper)*0.5
    q = 1-(1-confidence)/2
    sigma = (upper-mu)/norm.ppf(q)
    x = python_random.gauss(mu,sigma)
    
    if log:
        x = math.exp(x)
    return max(min(x,upper_bound),lower_bound)

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
    

    
