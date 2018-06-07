from learning2read.utils import LCS,better_match
import pickle
import os
from path import Path
from functools import reduce
import pandas as pd

class DataLoader: # htlin's data :)
    def __init__(self,path=r"/Users/qtwu/Downloads/data",verbose=True):
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

class DataMgr: # htlin's data :)
    def __init__(self,prefix,verbose=False):
        self.loader = DataLoader(prefix,verbose)
        self.verbose = verbose
    def __call__(self,keyword):
        return self.loader.load(keyword)

class PathMgr:
    def __init__(self,prefix,verbose=False):
        self.prefix = Path(prefix)
        self.verbose = verbose
    def __call__(self,fname=None):
        if not fname:
            return self.file_list()
        return self.prefix.joinpath(fname)
    def file_list(self):
        return os.listdir(self.prefix)
    def is_readable(self,fname):
        fname = self.prefix.joinpath(fname)
        result = os.access(fname,os.R_OK)
        
        return result


def save_pickle(fpath,data):
    with open(fpath,'wb') as f:
        pickle.dump(data,f)
        
def load_pickle(fpath):
    with open(fpath,'rb') as f:
        data=pickle.load(f)
    return data