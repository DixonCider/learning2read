# Hyperparameter Tuner Abstract Class
import abc

# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool

import os
from collections import defaultdict

import pandas as pd
import numpy as np
from learning2read.dnn import BatchNormDNN
from learning2read.io import save_pickle,load_pickle
from learning2read.utils import dict_to_code,RandomBox

from sklearn.neighbors.base import _get_weights, _check_weights
from sklearn.neighbors import KNeighborsRegressor
import datetime
now = datetime.datetime.now

class SklearnKNN(KNeighborsRegressor):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def normalize(self, x):
        return (x - self.xmean) / self.xstd
    def setup(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.xmean = np.mean(self.x, axis=0)
        self.xstd = np.std(self.x, axis=0) + 1 # zero safe
        self.x = self.normalize(self.x)
        self._init_params(
            n_neighbors=self.x.shape[1]+1,
            algorithm='kd_tree',
            leaf_size=30,
            metric='minkowski',
            p=2,
            metric_params=None,
            n_jobs=-1,
            **self.kwargs
        )
        self.weights = _check_weights('uniform')
#         self.weights = _check_weights('distance')
    def fit(self, x, y):
        self.setup(x,y)
        super(self.__class__, self).fit(self.x, self.y)
    def predict(self, X, only_mean=False):
#         X = check_array(X, accept_sparse='csr')
        X = np.array(X)
        X = self.normalize(X)
        neigh_dist, neigh_ind = self.kneighbors(X)
        weights = _get_weights(neigh_dist, self.weights)
        _y = self._y
        if _y.ndim == 1:
            _y = _y.reshape((-1, 1))

        if weights is None:
            y_pred = np.mean(_y[neigh_ind], axis=1)
        else:
            y_pred = np.empty((X.shape[0], _y.shape[1]), dtype=np.float64)
            denom = np.sum(weights, axis=1)
            for j in range(_y.shape[1]):
                num = np.sum(_y[neigh_ind, j] * weights, axis=1)
                y_pred[:, j] = num / denom
                
        if self._y.ndim == 1:
            y_pred = y_pred.ravel()
        if only_mean:
            return y_pred
        else:
            return y_pred, np.amin(neigh_dist, axis=1)
    def predict_df(self, X):
        mean, dist = self.predict(X)
        df = pd.DataFrame({'mean':mean, 'dist':dist})
        return df
    @classmethod
    def demo(cls):
        N, N2 = 12, 6
        x = [[1+t,2+t,3+t] for t in range(N)]
        y = [3*(t%3+1) for t in range(N)]
        xt= [[15-t,15-t,15-t] for t in range(N2)]
        model = cls()
        model.fit(x, y)
        return model.predict_df(xt)

class KNNOptimization:
    def __init__(self, param_list, target, form='df', param_only=True):
        assert form in ['df', 'list']
        self.model = SklearnKNN()
        self.target = target
        self.x_names = [d['name'] for d in param_list]
        self.param_list = param_list
        self.form = form
        self.param_only = param_only
        
    def advice(self, rdf, N=1000):
        rdf = rdf.dropna()
        self.model.fit(rdf.loc[:, self.x_names], rdf.loc[:, self.target])
        params = []
        for d in self.param_list:
            if d.get('const'):
                x = np.ones(N) * d['const']
            else:
                x = np.random.uniform(d['min'], d['max'], size=N)
            x = x.astype(d['type'])
            params.append({'name':d['name'], 'value':x})
        x_pred = np.vstack([d['value'] for d in params]).transpose()
        mean, dist = self.model.predict(x_pred)
        params.append({'name':'mean', 'value':mean})
        params.append({'name':'dist', 'value':dist})
        pdf = pd.DataFrame({d['name']:d['value'] for d in params})
        pdf.sort_values(['mean', 'dist'], ascending=[True, False], inplace=True)
        pdf = pdf.reset_index(drop=True)
        max_dist = 0
        used = []
        for i in range(pdf.shape[0]):
            r = pdf.iloc[i, :]
            if max_dist < r['dist']:
                max_dist = r['dist']
                used.append(1)
            else:
                used.append(0)
        cdf = pd.DataFrame({'used':used})
        result = pd.concat([pdf, cdf], axis=1)
        result = result.loc[result['used']==1, :]
        if self.param_only:
            result = result.drop(['mean', 'dist', 'used'], axis=1)
        if self.form=='list':
            result = result.to_dict('record')
        return result

class BaseTuner(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        return NotImplemented

    @abc.abstractmethod
    def tune(self):
        return NotImplemented

    @abc.abstractmethod
    def save(self):
        return NotImplemented
    
    @property
    def time_elapsed(self):
        try:
            return (now()-self.start).total_seconds()
        except:
            self.start = now()
            return 0
    
    # multiprocessing suffer from "XXX can't pickle" ... OTL
    # @property
    # def pool(self):
    #     try:
    #         self._pool
    #     except AttributeError:
    #         self._pool = Pool()
    #     return self._pool

class EpochBasedTuner(BaseTuner):
    class Model(BatchNormDNN):
        def init(self, data_dict):
            self.x, self.y = data_dict['x'], data_dict['y']
            self.nin = self.x.size(1)
            self.y = self.y.view(self.x.size(0), -1)
            self.nout = self.y.size(1)

            self.is_val_mode = True
            self.xv, self.yv = data_dict['xv'], data_dict['yv']
            self.yv = self.yv.view(self.xv.size(0), -1)

            # BatchNormDNN no normalize
            # self.normalize(zero_safe=True)
            super(self.__class__, self).init()

        def epoch(self):
            super(self.__class__, self).epoch(None)
            self.ein.append(float(self.loss_func(self.module(self.x), self.y)))
            self.eval.append(float(self.loss_func(self.module(self.xv), self.yv)))
            
    def __init__(self,
                 pid, data_list, fpath,
                 time_limit_per_tune=None, max_epochs=None,
                 verbose=True, resume=True):
        assert type(time_limit_per_tune)!=type(None) or type(max_epochs)!=type(None)
        self.pid = pid
        self.data_list = data_list
        self.samples = data_list[0]['x'].size(0)
        self.model_list = None
        self.fpath = fpath
        self.time_limit_per_tune = time_limit_per_tune or 1e+10
        self.max_epochs = max_epochs or 1e+10
        self.verbose = verbose
        self.rlist = []
        path = fpath(pid)
        if resume:
            if os.access(path, os.R_OK):
                self.rlist = load_pickle(path)
                if verbose:
                    print("target file %s detected"%path)
                    print("%d datas loaded"%len(self.rlist))
        self.rcache = defaultdict(lambda: -1)
        self.start = None
        
    def new_model(self,param):
        return self.__class__.Model(**param)
        
    @property
    def df(self):
        rdf = pd.DataFrame(self.rlist)
        rdf = rdf.sort_values('E_val')
        rdf = rdf.loc[:,[
            'E_in', 'E_in_std', 'E_val', 'E_val_std', 'best_iepoch', 
            'units', 'layers', 'learning_rate', 'time', 'nepoch', 'pid', 'samples']]
        return rdf
    
    def save(self):
        save_pickle(self.fpath(self.pid), self.rlist)
    
    def check_param(self, param):
        param_used = {}
        param_used.update(param)
        param_used.update({
            'batch_size': 1024,
        })
        param_used['units'] = int(param_used['units'])
        param_used['layers'] = int(param_used['layers'])
        return param_used
    
    def tune_init(self, param):
        # check param & cache
        param_used = self.check_param(param)
        pcode = dict_to_code(param_used)
        if self.rcache[pcode]>=0:
            return self.rlist[self.rcache[pcode]]
        
        # init models
        self.model_list = []
        K = len(self.data_list)
        for i in range(K):
            model = self.new_model(param_used)
            try:
                model.init(self.data_list[i])
            except: # MLE?
                result = {
                    'pcode' : -1,
                    'pid' : self.pid,
                }
                self.rlist.append(result)
                return result
            self.model_list.append(model)
        return None, param_used

    def tune(self, param):
        # init & handle exceptions such as MLE / identity param
        result_fail, param_used = self.tune_init(param)
        if result_fail:
            return result_fail
        pcode = dict_to_code(param_used)

        # train
        K = len(self.data_list)
        iepoch = 0
        self.start = now()
        while True:
            # sequentially train because python multiprocessing overhead
            try:
                for i in range(K):
                    self.model_list[i].epoch()
            except Exception as e:
                print(e)
                break
            if self.verbose:
                print("iepoch %d done. eval=%s"%(iepoch, str([round(self.model_list[i].last_eval,6) for i in range(5)])))
            if self.early_stop(iepoch):
                break
            if iepoch>3 and self.model_list[0].last_ein>1000:
                break
            iepoch += 1
        nepoch = iepoch+1
            
        # conclude K models information
        evals = [sum([m.eval[i] for m in self.model_list])/K for i in range(nepoch)]
        best_eval = min(evals)
        best_iepoch = 0
        for i in range(nepoch):
            if evals[i]==best_eval:
                best_iepoch = i
                break
        
        bein  = [m.ein[best_iepoch] for m in self.model_list]
        beval = [m.eval[best_iepoch] for m in self.model_list]
        result = {}
        result.update(param_used) # instead of "param"
        result.update({
            'time' : self.time_elapsed,
            'pcode' : pcode,
            'best_iepoch' : best_iepoch,
            'nepoch' : nepoch,
            'samples' : self.samples,
            'E_in' : np.mean(bein),
            'E_in_std' : np.std(bein),
            'E_val' : np.mean(beval),
            'E_val_std' : np.std(beval),
            'pid' : self.pid,
        })
        self.rcache[pcode] = len(self.rlist)
        self.rlist.append(result)
        return result
    
    def early_stop(self, iepoch):
        if iepoch > self.max_epochs:
            return True
        if self.time_elapsed > self.time_limit_per_tune:
            return True
        return False
    
    def random_param(self):
        R = RandomBox() # no seed is good seed
        return {
            'units'  : R.draw_int(3, 10 , 2, 50),
            'layers' : R.draw_int(3, 10 , 1, 50),
            'learning_rate' : R.draw_log(0.01, 0.1, 0.0001, 0.5),
        }

    def rs(self, n_iter):
        for i in range(n_iter):
            param = self.random_param()
            if self.verbose:
                print(param)
            result = self.tune(param)
            if self.verbose:
                print(result)
            self.save()
            
    @property
    def knn_optimizer(self):
        try:
            return self._knn_optimizer
        except:
            assert len(self.rlist)>=3+1
            self._knn_optimizer = KNNOptimization([
                {'name':'units', 'type':'int', 'min':2, 'max':200},
                {'name':'layers', 'type':'int', 'min':1, 'max':30},
                {'name':'learning_rate', 'type':'float', 'min':0.01, 'max':2},
            ], 'E_val')
            return self._knn_optimizer
    
    def knno(self, n_iter):
        for i in range(n_iter):
            advice = self.knn_optimizer.advice(self.df, 10000)
            print(advice)
            param = advice.sample(1).to_dict('records')[0]
            print(param)
            result = self.tune(param)
            print(result)
            self.save()