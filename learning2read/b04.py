# "runner" classes made by b04303128 :p
from learning2read.unsupervised import Pow2AutoEncoder
from learning2read.utils import alod
import torch
from collections import defaultdict
import pandas as pd
class BookVectorPow2AutoEncoder:
    """
        'class'  : 'learning2read.b04.BookVectorPow2AutoEncoder',
        'output' : 'book_vector',
        'input_data' : 'df_total',
        'domain_filter_num' : 2, # book with >=2 users
        'codomain_filter_num' : 1000, # user with >=1000 books
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