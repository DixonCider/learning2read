# "runner" classes made by b04303128 :p

from learning2read.unsupervised import Pow2AutoEncoder
from learning2read.utils import alod,draw,Index,IndexFold,list_diff,dict_to_code,RandomBox
from learning2read.io import PathMgr,DataMgr,save_pickle,load_pickle
from learning2read.proc import Procedure
from learning2read.preprocessing import RowFilter
from learning2read.dnn import SeluDNN
from multiprocessing import Pool
from bayes_opt import BayesianOptimization
import sys
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import datetime
now = datetime.datetime.now
import random
R = random.Random()

PATH_MAC={
    'data' : r"/Users/qtwu/Downloads/data",
    'cache' : r"/Users/qtwu/pickle/",
}
PATH_LIN2={
    'data' : r"/tmp2/b04303128/data",
    'cache' : r"/tmp2/b04303128/data/cache",
    'doc' : r"/home/student/04/b04303128/mlfinal",
}
File = PathMgr(PATH_MAC['cache'])
Data = DataMgr(PATH_MAC['data'])
File2 = PathMgr(PATH_LIN2['cache'])
Data2 = DataMgr(PATH_LIN2['data'])
Doc2 = PathMgr(PATH_LIN2['doc'])
Doc = Doc2

class SeluDNN_Tune(SeluDNN):
    def fit(self, time_limit):
        self.init()
        st = now()
        for iepoch in range(self.epochs):
            self.epoch(iepoch)
            self.epoch_end(iepoch)
            if (now()-st).total_seconds() > time_limit:
                self.epochs = iepoch+1
                break
        return self
    def setup(self, fold_dict):
        self.x = fold_dict['x']
        self.nin = self.x.size(1)
        self.y = fold_dict['y']
        self.y = self.y.view(self.x.size(0), -1)
        self.nout = self.y.size(1)
        self.is_val_mode = True
        self.xv = fold_dict['xv']
        self.yv = fold_dict['yv']
        self.yv = self.yv.view(self.xv.size(0), -1)
    def epoch(self, iepoch):
        for i,(x,y) in enumerate(self.dataloader):
            pred = self.module(x)
            loss = self.loss_func(pred, y)
            
            print("iepoch[%2d] mini-batch %5d : %15.4f"%(iepoch,i,float(loss)), file=sys.stderr, end='\r')
            sys.stderr.flush()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print("iepoch[%2d] optimizer done."%(iepoch), file=sys.stderr, end='\r')
            sys.stderr.flush()
class SeluDNN_Tuner:
    def __init__(self, pid, K_fold, time_limit, epochs_fixed, fload, fsave, fmodel):
        self.pid = pid
        self.K_fold = K_fold
        self.time_limit = time_limit
        self.epochs_fixed = epochs_fixed
        self.fold = [fload(i) for i in range(K_fold)]
        self.fsave = fsave
        self.fmodel = fmodel
        self.rlist = []
    def tune(self, param):
        param.update({'epochs':self.epochs_fixed})
        K = self.K_fold
        result = {}
        result.update(param)
        def f(i_fold):
            model = self.fmodel(param)
            model.setup(self.fold[i_fold])
            model.fit(self.time_limit)
            return model
            
        # train
        st = now()
        models = [f(i) for i in range(K)]
        tcost = (now()-st).total_seconds()
        
        # conclude K models information
        real_epochs = [m.epochs for m in models]
        evals = [sum([m.eval[i] for m in models])/K for i in range(min(real_epochs))]
        best_eval = min(evals)
        best_epochs = 0
        for i in range(min(real_epochs)):
            if evals[i]==best_eval:
                best_epochs = i
                break
        
        bein  = [m.ein[best_epochs] for m in models]
        beval = [m.eval[best_epochs] for m in models]
        result.update({
            'time' : tcost,
            'pcode' : dict_to_code(param),
            'best_epochs' : best_epochs,
            'E_in' : np.mean(bein),
            'E_in_std' : np.std(bein),
            'E_val' : np.mean(beval),
            'E_val_std' : np.std(beval),
        })
        for i in range(K):
            result['epo%d'%i] = real_epochs[i]
        return result

    @property
    def df(self):
        return pd.DataFrame(self.rlist)

    def save(self):
        return self.fsave(self.pid, self.rlist)

    def rs(self, total_time): # randomized search
        R = RandomBox() # no seed is good seed
        st = now()
        while (now()-st).total_seconds()<total_time:
            param = {
                'units' : R.draw_int(3,9,1,15),
                'layers' : R.draw_int(3,9,1,15),
                'learning_rate' : R.draw_log(0.01,0.1,1e-5,0.5),
            }
            print(param)
            result = self.tune(param)
            print(result)
            self.rlist.append(result)
            self.save()
    def bo(self, total_time=30, kappa=1, init_points=3): # Bayesian Optimization
        st = now()
        cache = defaultdict(lambda:None)
        def target(**param):
            param['units'] = int(round(param['units']))
            param['layers'] = int(round(param['layers']))
            pcode = dict_to_code(param)
            if cache[pcode]:
                print("cache!")
                return -cache[pcode]
            result = self.tune(param)
            
            print("remain = %10.2f"%(float(total_time - (now()-st).total_seconds())), file=sys.stderr, end='\r')
            sys.stderr.flush()
            self.rlist.append(result)
            self.save()
            cache[pcode] = result['E_val']
            return -result['E_val']
        obj = BayesianOptimization(target, {
            'units' : (1,30),
            'layers' : (1,30),
            'learning_rate' : (0.005,0.5),
        })
        obj.maximize(init_points=init_points, n_iter=0, acq='ucb', kappa=kappa)
        while (now()-st).total_seconds()<total_time:
            obj.maximize(init_points=0, n_iter=1, acq='ucb', kappa=kappa)

class ProcValidation:
    """
    'class' : 'learning2read.b04.ProcValidation',
    'output' : 'P',
    'input_data' : 'raw_dataset',
    'proc' : proc40072, # run_id(0) -> hold y_valid -> run all
    'K_fold' : 5,
    'i_fold' : 0,
    'seed_fold' : 1,
    """
    @classmethod
    def run(cls,input_data,proc,i_fold,K_fold,seed_fold,verbose=True):
        P = Procedure(proc, verbose)
        P.load_data(input_data)
        P.run_id(0)
        assert type(P.var['df_total'])
        def validation_set(P, i, K, seed):
            df = P.var['df_total']
            df_train = df.loc[df['Book-Rating']>0, :]
            N = df_train.shape[0]
            idx_valid = IndexFold(K, N, seed)[i]
            idx_valid = df_train.iloc[idx_valid, :].index
            y_valid = df.loc[idx_valid, 'Book-Rating']
            df_sub = df.iloc[:, :]
            df_sub.loc[idx_valid, 'Book-Rating'] = -2
            P.var['df_total'] = df_sub.iloc[:, :]
            return (P, y_valid)
        P, y_valid = validation_set(P, i_fold, K_fold, seed_fold)
        P.run()
        assert type(P.var['df_total_features'])
        df_train = RowFilter.run(P.var['df_total_features'],r"lambda df:df['Book-Rating']>0",)['output']
        df_valid = RowFilter.run(P.var['df_total_features'],r"lambda df:df['Book-Rating']==-2",)['output']
        df_valid.loc[y_valid.index, 'Book-Rating'] = y_valid
        P.var['df_train'] = df_train
        P.var['df_valid'] = df_valid
        return {'output' : P}

class FileGen: # workstation Pool().map_aync(f,range(5))
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    def cv_preapre(self, pid):
        assert type(self.save)==type(lambda:None)
        raw_dataset = self.raw_dataset
        proc = self.proc
        K_fold = self.K_fold
        i_fold = pid
        seed_fold = self.seed_fold
        P = ProcValidation.run(raw_dataset,proc,i_fold,K_fold,seed_fold)['output']
        self.save(pid, P.var)
        return P
    def tune(self, pid, T=1, tbound=60, pgen=(lambda:{
                    'num_leaves' : int(draw(15,127,3,1023)),
                    'learning_rate' : draw(1e-2,1e-1,1e-10,1,log=True),
                    'n_estimators' : int(draw(100,200,4,400)),
                    'min_child_samples' : int(draw(15,63,7,1023)),
                }) ):
        save = self.save
        load = self.load
        check = self.check
        K_fold = self.K_fold
        assert type(self.save)==type(lambda:None)
        assert type(self.check)==type(lambda:None)
        assert type(self.load)==type(lambda:None)

        # prepare df_train, df_valid
        df_train = []
        df_valid = []
        for i_fold in range(K_fold):
            print("loading %d/%d fold"%(i_fold,K_fold))
            assert check(i_fold)
            fold = load(i_fold)
            df_train.append(fold['df_train'])
            df_valid.append(fold['df_valid'])
        
        # tune
        rlist = []
        for ti in range(T):
            try:
                param = pgen()
                print(param)
                for i_fold in range(K_fold):
                    st = now()
                    result = LightGBMRegressor.run([df_train[i_fold], df_valid[i_fold]], param)
                    tcost = (now()-st).total_seconds()
                    result.update({
                        'ti' : ti,
                        'pcode' : dict_to_code(param),
                        'i_fold' : i_fold,
                        'K_fold' : K_fold,
                        'time' : tcost,
                    })
                    result.update(param)
                    rlist.append(result)
                    print('i_fold = %d\nresult = %s'%(i_fold,str(result)))
                    if tcost > tbound:
                        print("cut param run time too long")
                        break

                # save rlist every round
                try:
                    save(pid, rlist)
                except Exception as e:
                    print("[WARNING] save failed !!!!")
                    print(e)
            except Exception as e:
                print("[WARNING] ti=%d failed."%ti)
                print(e)


class LightGBMRegressor:
    """
    input_data = [train, valid] -> validation mode
    input_data = train -> training mode
    """
    @classmethod
    def run(cls, input_data, param): # input_data = [df_train_v, df_valid]
        if type(input_data)==list:
            assert len(input_data)==2
            is_val_mode = True
            x_train = input_data[0].iloc[:, 1:]
            y_train = np.ravel(input_data[0].iloc[:, :1])
            x_valid = input_data[1].iloc[:, 1:]
            y_valid = np.ravel(input_data[1].iloc[:, :1])
        else:
            is_val_mode = False
            x_train = input_data.iloc[:, 1:]
            y_train = np.ravel(input_data.iloc[:, :1])
        seed = 1
        if param.get('seed'):
            seed = param['seed']
        lgb_param={
            'seed' : seed,
            'bagging_seed' : seed,
            'feature_fraction_seed' : seed,
            'drop_seed' : seed,
            'data_random_seed' : seed,
        }
        lgb_param.update({
            'objective' : 'regression_l1',
            'boosting_type' : 'gbdt',
            'num_leaves' : 31,
            'learning_rate' : 0.1,
            'n_estimators' : 100,
            'min_child_samples' : 20,
            'n_jobs' : -1,
            'tree_learner' : 'data',
        })
        lgb_param.update(param)
        model = lgb.LGBMRegressor(**lgb_param)
        model.fit(x_train, y_train)

        E_in, E_val = None, None

        try:
            E_in = mean_absolute_error(y_train, model.predict(x_train))
        except:
            pass
            
        if is_val_mode:
            try:
                E_val = mean_absolute_error(y_valid, model.predict(x_valid))
            except:
                pass
            model = None
        return {
            'output' : model,
            'E_in' : E_in,
            'E_val' : E_val,
        }

class LightGBMRegressorTuner:
    def __init__(self, df_train, K=5, fold_seed=999, verbose=False):
        self.result_list = []
        self.K = K
        self.N = df_train.shape[0]
        self.fold = IndexFold(K, self.N, fold_seed)
        self.verbose = verbose
        self.df_train = df_train
    @property
    def df(self):
        return pd.DataFrame(self.result_list)
    def run_param(self, param):
        N = self.df_train.shape[0]
        K = self.K
        for i in range(K):
            if self.verbose:
                print(i,K)
            idx_valid = self.fold[i]
            idx_train = list_diff(range(N), idx_valid)
            df_valid = self.df_train.iloc[idx_valid, :]
            df_train_v = self.df_train.iloc[idx_train, :]
            st = now()
            result = LightGBMRegressor.run([df_train_v, df_valid], param)
            t_cost = (now() - st).total_seconds()
            result.update({
                'pcode' : dict_to_code(param),
                'fold_i' : i,
                'fold_k' : K,
                'time' : t_cost,
            })
            result.update(param)
            self.result_list.append(result)
            if self.verbose:
                print(result)
        if self.verbose:
            print(self.df)
        return self
    @classmethod
    def tune1(cls,df_train,T=100,name="tune1",path_mgr=None):
        obj = cls(df_train)
        for ti in range(T):
            try:
                param = {
                    'num_leaves' : int(draw(15,127,3,1023)),
                    'learning_rate' : draw(1e-3,1e+2,1e-10,1e+5,log=True),
                    'n_estimators' : int(draw(50,500,1,1000)),
                    'min_child_samples' : int(draw(15,63,7,1023)),
                }
                print(param)
                obj.run_param(param)
            except:
                print("ti=%d failed."%ti)
            if ti%5==0 or ti==T-1:
                if path_mgr:
                    fname = "%s_%d"%(name,T)
                    print(fname)
                    save_pickle(path_mgr(fname), obj)
        return obj

class LightGBMRandomForest:
    """
        'output' : 'model_rf', # a sklearn like model with (fit,predict)
        'input_data' : 'df_train',
        'param' : {
            'random_state' : 1,
        }
    """
    @classmethod
    def run(cls,input_data,param):
        # load param
        _param = param.copy() # copy before delete
        seed = _param['random_state']
        lgb_param={
            'seed' : seed,
            'bagging_seed' : seed,
            'feature_fraction_seed' : seed,
            'drop_seed' : seed,
            'data_random_seed' : seed,
        }
        del _param['random_state']
        lgb_param.update(_param)
        lgb_param.update({
            'objective' : 'regression_l1',
            "boosting" : "rf",
            'num_leaves' : 300,
            'max_depth' : -1,
            'learning_rate' : 0.01,
            "min_child_samples" : 20,
            "feature_fraction" : 0.4,
            "bagging_freq" : 10,
            "bagging_fraction" : 0.4 ,
            "bin_construct_sample_cnt" : 200000,
        })

        # load data
        is_val_mode = False
        x_val, y_val, E_val = None, None, None
        if type(input_data)==list:
            # validation mode
            assert len(input_data)==2 # must input [train, val]
            x_val = input_data[1].iloc[:,1:]
            y_val = np.ravel(input_data[1].iloc[:,:1])
            train_data = input_data[0]
            is_val_mode = True
        else:
            train_data = input_data
        x_train = train_data.iloc[:,1:]
        y_train = np.ravel(train_data.iloc[:,:1])
        model = lgb.LGBMRegressor(**lgb_param)
        model.fit(x_train, y_train)
        E_in = model.score(x_train, y_train)
        if is_val_mode:
            E_val = model.score(x_val, y_val)
        # E_val= None
        return {
            'output' : model,
            'E_in' : E_in,
            'E_val' : E_val,
        }

"""
import lightgbm as lgb
# random forest mode
param = {}
model=lgb.LGBMRegressor(
    objective='regression_l1',
    **param
)
model.fit(x_train, y_train)
model.score(x_train, y_train) # best: 0.33981505052378735
"""

class BookVectorPow2AutoEncoder:
    """
        'class'  : 'learning2read.b04.BookVectorPow2AutoEncoder',
        'output' : 'book_vector',
        'input_data' : 'df_total',
        'domain_filter_num' : 2, # book with >=2 users
        'codomain_filter_num' : 400, # user with >=400 books
        'param' : {
            'code_length' : 16, 
            'activation' : 'SELU', 
            'solver' : 'Adam', 
            'learning_rate' : 0.01,
            'epochs' : 10,
            'random_state' : 1,
        },
    """
    @classmethod
    def run(cls,input_data,domain_filter_num,codomain_filter_num,param,
            domain="ISBN",codomain="User-ID",code_col_name_prefix='bv'):
        code_length = param['code_length']
        dfdo = input_data.groupby(domain).count()[codomain].sort_values(ascending=False)
        dfdo = dfdo.loc[dfdo>=domain_filter_num]
        dfco = input_data.groupby(codomain).count()[domain].sort_values(ascending=False)
        dfco = dfco.loc[dfco>=codomain_filter_num]
        
        do_to_id = defaultdict(lambda: -1) # dict[ISBN] = book_index
        co_to_id = defaultdict(lambda: -1) # dict[User-ID] = user_index
        print("input dim: ",[len(dfdo),len(dfco)])
        for i in range(len(dfdo)):
            do_to_id[dfdo.index[i]] = i
        for i in range(len(dfco)):
            co_to_id[dfco.index[i]] = i
        
        idx_list = [] # for torch.sparse.FloatTensor
        for r in alod(input_data):
            doid = do_to_id[r[domain]]
            coid = co_to_id[r[codomain]]
            if doid>=0 and coid>=0:
                idx_list.append([doid, coid])

        if len(idx_list)==0:
            print("[BookVectorPow2AutoEncoder] WARNING idx_list got nothing (too extreme filter?)")
            return {'output':pd.DataFrame(dfdo.index, columns=[domain])}

        print(torch.tensor(idx_list).t())

        idx_tns = torch.LongTensor(idx_list).t()
        val_tns = torch.ones(idx_tns.size(1))
        train_tns = torch.sparse.FloatTensor(idx_tns, val_tns, torch.Size([len(dfdo),len(dfco)]))
        train_tns = train_tns.to_dense()

        model = Pow2AutoEncoder(**param)
        model.fit(train_tns)
        code = model.predict(train_tns)
        zero_vec = model.predict(torch.zeros(len(dfco)))
        vec_lst = []
        for do_df_id in dfdo.index: # domain = 'ISBN' = do_df_id
            doid = do_to_id[do_df_id]
            # print(do_df_id,"mapsto",doid)
            if doid<0: # not in model
                vec_lst.append(zero_vec)
            else:
                vec_lst.append(code[doid])
        
        output = pd.concat([
            pd.DataFrame(dfdo.index, columns=[domain]),
            pd.DataFrame(vec_lst, columns=["%s%d"%(code_col_name_prefix,i+1) for i in range(code_length)])],
            axis=1)

        return {'output' : output}

class UserVectorPow2AutoEncoder:
    @classmethod
    def run(cls,**kwargs):
        R=BookVectorPow2AutoEncoder.run(
                domain="User-ID",
                codomain="ISBN",
                code_col_name_prefix='uv',
                **kwargs)
        return R
        # return {}.update(BookVectorPow2AutoEncoder.run(
        #         domain="User-ID",
        #         codomain="ISBN",
        #         code_col_name_prefix='uv',
        #         **kwargs))