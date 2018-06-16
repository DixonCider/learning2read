# Deep Neural Network
import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
import sys
import datetime
now = datetime.datetime.now

class SeluDNNModule(nn.Module):
    def __init__(self, features, units, layers, outputs, init_seed=1):
        super(SeluDNNModule, self).__init__()
        torch.manual_seed(init_seed)
        layer_list = []
        for li in range(layers):
            n_in  = units if li else features
            n_out = units
            L = nn.Linear(n_in, n_out)
            nn.init.normal_(L.weight.data, 0, n_in ** -0.5)
            layer_list.append(L)
            layer_list.append(nn.modules.activation.SELU(True))
        layer_list.append(nn.Linear(units, outputs))
        self.dnn = nn.Sequential(*layer_list)
    def forward(self,x):
        return self.dnn(x)

class SeluDNN:
    """
    activation function : SELU
    init : LeCun normal ; uniform (output layer)
    optimizer : Adam
    loss function : L1Loss
    dropout : 0
    """
    def __init__(
        self,units,layers,learning_rate,epochs,
        batch_size=32,seed=1,init_seed=1,
        verbose=False):
        self.units = units
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        # self.time_limit = time_limit
        self.seed = seed
        self.init_seed = init_seed
        self.verbose = verbose
        self.x = None
        self.y = None
        self.xv = None
        self.yv = None
        self.ein = []
        self.eval = []
        # self.result = {'record':[]}
        self.clear_torch()
    def clear_torch(self):
        self._module = None
        self._loss_func = None
        self._optimizer = None
        self._dataloader = None
        self._train_dataset = None
        return self


    def fit(self,x_trian,y_train,x_valid=None,y_valid=None):
        self.setup_train(x_trian, y_train)
        self.setup_valid(x_valid, y_valid)
        
        # set initial weight by init_seed
        self.module

        # training seed, controls mini-batch / dropout
        torch.manual_seed(self.seed) 

        for iepoch in range(self.epochs):
            self.epoch(iepoch)
            self.epoch_end(iepoch)
            if self.need_early_stop():
                break
        return self

    def epoch(self, iepoch):
        for x,y in self.dataloader:
            pred = self.module(x)
            loss = self.loss_func(pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def epoch_end(self, iepoch):
        self.ein.append(float(self.loss_func(self.module(self.x), self.y)))
        if self.is_val_mode:
            self.eval.append(float(self.loss_func(self.module(self.xv), self.yv)))
        if self.verbose:
            print("iepoch = %5d  Ein = %12.5f  Eval = %12.5f"%(iepoch, self.ein[-1], self.eval[-1]), file=sys.stderr, end="\r")
            sys.stderr.flush()

    def predict(self,x):
        pass
        
    def need_early_stop(self):
        return False
        
    def setup_train(self,x_trian,y_train):
        self.x = torch.FloatTensor(np.array(x_trian))
        self.nin = self.x.size(1)
        self.y = torch.FloatTensor(np.array(y_train))
        self.y = self.y.view(self.x.size(0), -1)
        self.nout = self.y.size(1)
        return self

    def setup_valid(self,x_valid,y_valid):
        self.is_val_mode = type(x_valid)!=type(None) and type(y_valid)!=type(None)
        if not self.is_val_mode:
            return self
        self.xv = torch.FloatTensor(np.array(x_valid))
        self.yv = torch.FloatTensor(np.array(y_valid))
        self.yv = self.yv.view(self.xv.size(0), -1)
        return self

    def __call__(self,x):
        return self._module(x)
    
    @property
    def module(self):
        if not self._module:
            self._module = SeluDNNModule(
                self.nin, self.units, self.layers, self.nout,
                init_seed=self.init_seed)
        return self._module

    @property
    def dataloader(self):
        if not self._dataloader:
            self._dataloader = Data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True)
        return self._dataloader

    @property
    def train_dataset(self):
        if not self._train_dataset:
            self._train_dataset = Data.TensorDataset(self.x, self.y)
        return self._train_dataset

    @property
    def optimizer(self):
        if not self._optimizer:
            self._optimizer = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        return self._optimizer
        
    @property
    def loss_func(self):
        if not self._loss_func:
            self._loss_func = torch.nn.L1Loss()
        return self._loss_func


# import torch
# from torch.autograd import Variable
# import torch.nn.functional as F
# # from sklearn.utils import check_array

# def check_array(array_like_object):
#     return np.array(array_like_object)
# def check_tensor(array_like_object):
#     return torch.from_numpy(array_like_object).float() # cause GPU does float faster
# def check_tensor_array(array_like_object):
#     return check_tensor(check_array(array_like_object))
# class DEV_MODULE(torch.nn.Module):
#     def __init__(self, n_feature, n_output, build_param):
#         super(DEV_MODULE, self).__init__()
#         self.units = build_param['units']
#         self.layers = build_param['layers']
#         self.activation = eval("F.%s"%build_param['activation'])
        
#         n_hidden_out = self.units
#         for i in range(self.layers):
#             n_hidden_in = self.units if i>0 else n_feature
# #             self['hidden%03d'%(i+1)] = torch.nn.Linear(n_hidden_in, n_hidden_out)
#             self.add_module('hidden%03d'%(i+1), torch.nn.Linear(n_hidden_in, n_hidden_out))
#         self.predict = torch.nn.Linear(n_hidden_out, n_output)

#     def forward(self, x):
#         for i in range(self.layers):
#             # x(l=3,i) = relu(s(l=3,i)) , with s(l=3,i)=[w(l=3)][x(l=2)]
#             x = self.activation(self.__getattr__('hidden%03d'%(i+1))(x))
#         x = self.predict(x) # linear output
#         return x

# class DEV:
#     def __init__(self,layers=4,units=3,activation='relu',learning_rate=0.1):
#         self.module = None # build when "fit" is called
#         self.layers = layers
#         self.units = units
#         self.activation = activation
#         self.learning_rate=learning_rate
#     def fit(self,x_train,y_train,iters=3):
#         x_train = check_array(x_train)
#         y_train = check_array(y_train)
#         self.x_tensor = Variable(check_tensor(x_train))
#         n_sample = self.x_tensor.size(0)
#         self.x_tensor = self.x_tensor.view(n_sample, -1)
#         n_feature = self.x_tensor.size(1)
#         self.y_tensor = Variable(check_tensor(y_train).view(n_sample,-1))
#         n_output = self.y_tensor.size(1)
#         self.module=DEV_MODULE(n_feature=n_feature,
#                                n_output=n_output,
#                                build_param=self.__dict__)
#         self.optimizer = torch.optim.Adam(self.module.parameters(),
#                                     lr=self.learning_rate)
#         self.loss_func = torch.nn.L1Loss()  # this is for regression mean squared loss
        
#         for itr in range(iters):
#             self.prediction = self.module(self.x_tensor)     # input x and predict based on x

#             self.loss = self.loss_func(self.prediction, self.y_tensor)     # must be (1. nn output, 2. target)

#             self.optimizer.zero_grad()   # clear gradients for next train
#             self.loss.backward()         # backpropagation, compute gradients
#             self.optimizer.step()        # apply gradients
#             print(itr,self.loss)
            
#         return self
#     def predict(self,x_test):
#         x_test_tensor = check_tensor_array(x_test)
#         prediction = self.module(x_test_tensor)
#         return prediction.data.numpy()
        
# model = DEV(28,6,'selu',learning_rate=0.01)
# model.fit(x_train,y_train,50) # best ein=1.4511