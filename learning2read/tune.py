# Hyperparameter Tuner Abstract Class
import abc

# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool

import pandas as pd
import numpy as np

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
    pass