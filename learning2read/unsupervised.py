# unsupervised
import torch
from torch.autograd import Variable
import torch.nn as nn

from learning2read.utils import check_tensor_array
import datetime
now=datetime.datetime.now

def pow2_list(a,b):
    """
    30 features -> length 2 code
    encoder net: 30->32->16->8->4->2--code--
    decoder net: --code--2->4->8->16->32->30
    pow2_list(30,2) = [4,8,16,32]
    """
    assert a>b
    m = 1
    while m<=b:
        m *= 2
    rlist = [m] # want m>b
    while m<=a:
        m *= 2
        rlist.append(m) # want m<=2a
    return rlist

class Pow2AutoEncoderModule(nn.Module):
    def __init__(self, hyper_param):
        super(Pow2AutoEncoderModule, self).__init__()
        encoder_units_list = []
        decoder_units_list = []
        p2l = pow2_list(hyper_param['x'].size(1),hyper_param['code_length'])
        for i in range(len(p2l)):
            if i==0:
                decoder_units_list.append([hyper_param['code_length'], p2l[i]])
            else:
                decoder_units_list.append([p2l[i-1], p2l[i]])
        decoder_units_list.append([p2l[-1], hyper_param['x'].size(1)])
        for upair in reversed(decoder_units_list):
            encoder_units_list.append([upair[1],upair[0]])
            
        encoder_layers = []
        decoder_layers = []
        for i,upair in enumerate(encoder_units_list):
            if i>0:
                encoder_layers.append(eval("nn.modules.activation.%s(True)"%hyper_param['activation']))
            encoder_layers.append(nn.Linear(upair[0], upair[1]))
        self.encoder = nn.Sequential(*encoder_layers)
        for i,upair in enumerate(decoder_units_list):
            if i>0:
                decoder_layers.append(eval("nn.modules.activation.%s(True)"%hyper_param['activation']))
            decoder_layers.append(nn.Linear(upair[0], upair[1]))
        decoder_layers.append(nn.modules.activation.ReLU(True))
        self.decoder = nn.Sequential(*decoder_layers)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Pow2AutoEncoder:
    def __init__(self,
                 code_length=2,
                 activation='ReLU' or 'SELU',
                 solver='Adam',
                 learning_rate=0.1,
                 random_state=1,
                 epochs=5):
        self.__dict__.update(locals())
        del self.self # fix locals
        self.module = None # build when "fit" is called
    def fit(self, x_train, epochs=None): # x_train sparse
        epochs = epochs or self.epochs
        torch.manual_seed(self.random_state)
        if not isinstance(x_train,torch.Tensor):
            x_train = check_tensor_array(x_train)
        self.x = Variable(x_train)
        self.y = Variable(x_train) # AutoEncoder :)
        self.module=Pow2AutoEncoderModule(self.__dict__)
        optimizer = eval("torch.optim.%s"%self.solver)(self.module.parameters(),lr=self.learning_rate)
        loss_func = torch.nn.MSELoss()
        
        st=now()
        for epoch in range(epochs):
            pred = self.module(self.x)
            loss = loss_func(pred, self.y)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            print(epoch,loss,now()-st)
        return self
    def predict(self,x_test):
        if not isinstance(x_test,torch.Tensor):
            x_test = check_tensor_array(x_test)
        prediction = self.module.encoder(x_test)
        return prediction.data.numpy()