# Deep Neural Network
import numpy as np
import torch
import torch.utils.data as TUData
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
        self.start = None
        self.is_val_mode = False
        self.normalized = False
        # self.result = {'record':[]}
        self.init_torch()
    def init_torch(self):
        self.gpu_count = torch.cuda.device_count()
        self.device = torch.device("cpu") if self.gpu_count==0 else torch.device("cuda:0")
        if self.verbose:
            print("[SeluDNN] gpu_count = %d"%self.gpu_count)
            print("[SeluDNN] device = %s"%self.device)
        self._module = None
        self._loss_func = None
        self._optimizer = None
        self._dataloader = None
        self._train_dataset = None
        return self

    def init(self):
        # optional
        # self.init_torch()

        # set initial weight by init_seed
        self.module

        # training seed, controls mini-batch / dropout
        torch.manual_seed(self.seed)

        self.start = now()
        

    def fit(self, x_trian=None, y_train=None, x_valid=None, y_valid=None):
        self.setup_train(x_trian, y_train)
        self.setup_valid(x_valid, y_valid)
        self.init()
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
            if self.is_val_mode:
                print("iepoch = %5d [%8.2f] Ein = %12.5f  Eval = %12.5f"%(iepoch, self.time_elapsed, self.ein[-1], self.eval[-1]), file=sys.stderr, end="\r")
            else:
                print("iepoch = %5d [%8.2f] Ein = %12.5f"%(iepoch, self.time_elapsed, self.ein[-1]), file=sys.stderr, end="\r")
            sys.stderr.flush()

    def predict(self,x):
        x = np.array(x)
        x = self.as_tensor(x)
        y = self.module(x)
        # if self.normalized:
            # y = self.ymean + self.ystd * y
        return y
        
    def need_early_stop(self):
        return False
        
    def setup_train(self,x_train,y_train):
        if type(x_train)==type(None) or type(y_train)==type(None):
            assert type(self.x)!=type(None)
            assert type(self.y)!=type(None)
            return self # do nothing
        self.x = self.as_tensor(np.array(x_train))
        self.nin = self.x.size(1)
        self.y = self.as_tensor(np.array(y_train))
        self.y = self.y.view(self.x.size(0), -1)
        self.nout = self.y.size(1)
        return self

    def setup_valid(self,x_valid,y_valid):
        if type(x_valid)==type(None) or type(y_valid)==type(None):
            return self
        self.is_val_mode = True
        self.xv = self.as_tensor(np.array(x_valid))
        self.yv = self.as_tensor(np.array(y_valid))
        self.yv = self.yv.view(self.xv.size(0), -1)
        return self
    
    # model related utils
    def normalize(self, zero_safe=False):
        # x, (x-x.mean(dim=0)), x.std(dim=0)
        self.xmean = self.x.mean(dim=0)
        self.xstd = self.x.std(dim=0) + int(zero_safe)
        # self.ymean = self.y.mean(dim=0)
        # self.ystd = self.y.std(dim=0) + int(zero_safe)
        self.x = (self.x-self.xmean) / self.xstd
        # self.y = (self.y-self.ymean) / self.ystd
        if self.is_val_mode: # validation set normalize by training set
            self.xv = (self.xv-self.xmean) / self.xstd
            # self.yv = (self.yv-self.ymean) / self.ystd
        self.normalized = True

    def as_tensor(self, data):
        return torch.FloatTensor(data).to(self.device)
    @property
    def time_elapsed(self):
        if not self.start:
            self.start = now()
        return (now()-self.start).total_seconds()

    # pytorch wrappers
    def __call__(self,x):
        return self._module(x)
    
    @property
    def module(self):
        if not self._module:
            self._module = SeluDNNModule(
                self.nin, self.units, self.layers, self.nout,
                init_seed=self.init_seed).to(self.device)
        return self._module

    @property
    def dataloader(self):
        if not self._dataloader:
            self._dataloader = TUData.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True)
        return self._dataloader

    @property
    def train_dataset(self):
        if not self._train_dataset:
            self._train_dataset = TUData.TensorDataset(self.x, self.y)
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
