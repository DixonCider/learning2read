from difflib import ndiff
from functools import reduce
from path import Path # package "path.py"
import pandas as pd
import numpy as np
import scipy

def LCS(s1,s2): # length of LCS(s1,s2)
    diff=ndiff(s1,s2)
    result=0
    for part in diff:
        if part[0]==' ':
            result+=1
    return result
    

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
            lambda a,b:a if DataLoader.better_match(keyword,a,b) else b,
            self.file_list)
        fpath=Path(self.path).joinpath(filename_matched)
        if self.verbose:
            print(fpath)
        return pd.read_csv(fpath,**kwargs)
    
    @staticmethod
    def better_match(key,s1,s2):
        l1=LCS(key,s1)
        l2=LCS(key,s2)
        if l1==l2:
            return len(s1)<len(s2)
        return l1>l2
