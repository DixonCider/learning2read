# unsupervised
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

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
class VariationalAutoEncoderModule(nn.Module):
    def __init__(self, hyper_param):
        super(VariationalAutoEncoderModule, self).__init__()
        encoder_units_list = []
        decoder_units_list = []
        # Contruct the [#ofInput, #ofOutput] as list for encoder and deocder.
        p2l = pow2_list(hyper_param['x'].size(1),hyper_param['code_length'])
        # Missing _ to code_length layer
        for i in range(len(p2l)-1):
            decoder_units_list.append([p2l[i], p2l[i+1]])
        decoder_units_list.append([p2l[-1], hyper_param['x'].size(1)])
        for upair in reversed(decoder_units_list):
            encoder_units_list.append([upair[1], upair[0]])
        encoder_layers = []
        decoder_layers = []
        # Append to encoder.
        for i, upair in enumerate(encoder_units_list):
            # TODO : add the nn layer here.
            if i>0:
                encoder_layers.append(eval("nn.modules.activation.%s(True)"%hyper_param['activation']))
            encoder_layers.append(nn.Linear(upair[0], upair[1]))
        # Unpack all layers in encoder_layers to sequential.
        self.encoder = nn.Sequential(*encoder_layers)
        # Add the final Gaussain mu and var.
        self.encoderMu = nn.Linear(encoder_units_list[-1][1], hyper_param['code_length'])
        self.encoderLogVar = nn.Linear(encoder_units_list[-1][1], hyper_param['code_length'])
        # Append to decoder.
        # Append the bottom layer of decoder reparameterized from encoder.
        decoder_layers.append(nn.Linear(hyper_param['code_length'], hyper_param['code_length']*2))
        decoder_layers.append(nn.modules.activation.ReLU(True))
        for i, upair in enumerate(decoder_units_list):
            if i>0:
                decoder_layers.append(eval("nn.modules.activation.%s(True)"%hyper_param['activation']))
            decoder_layers.append(nn.Linear(upair[0], upair[1]))
        # Append final activation.
        decoder_layers.append(nn.modules.activation.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h1 = self.encoder(x)
        return self.encoderMu(h1), self.encoderLogVar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 117), size_average=False)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
        

class VariationalAutoEncoder:
    def __init__(
        self, 
        code_length=8, 
        activation='ReLU' or 'SELU',
        solver='Adam',
        learning_rate=0.01,
        random_state=1126,
        batch_size=128,
        cuda=False,
        log_interval=100):
        """
        Input {
            code_length : encoding code lendth (default=8)
            activation (ReLU or SeLU)
            solver (default=Adam)
            learning_rate (default=0.01)
            random_state : random seed (default=1126)
            batch_size (default=128)
            cuda : use cuda or not (default=False)
            log_interval : interval to log training progress (default=100)
        }
        """
        self.code_length = code_length
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.epochs = None
        self.model = None
        self.train_loader = None
        self.device = torch.device("cuda" if cuda else "cpu")
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    
    # Must beTensor before fit.
    def fit(self, x_train, epochs=1):
        """
        !! x_train must be preprocessed by b05.preprocessingForVAE !!
        Input {
            x_train : preprocessed x_train in torch.tensor form
            epochs : number of epochs
        }
        """
        self.epochs = epochs
        torch.manual_seed(self.random_state)
        # Transform x_train to tensor assuming preprocessed.
        # Please refer to 5001 for more detail.
        self.train_loader = torch.utils.data.DataLoader(
            BookVectorData(x_train),
            batch_size=self.batch_size, shuffle=True, **self.kwargs)
        # Add hyper_param
        hyper_param = {
            'training': True,
            'code_length': self.code_length,
            'activation': self.activation,
            'x': x_train
        }
        self.model = VariationalAutoEncoderModule(hyper_param).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

        # Training.
        for epoch in range(1,self.epochs+1):
            train_loss = 0
            self.model.train()
            for batch_idx, data in enumerate(self.train_loader):
                optimizer.zero_grad()
                # recon_batch is the autoencoded + resontructed data.
                recon_batch, mu, logvar = self.model(data)
                loss = self.model.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(data)))
            print('====> Epoch: {} Average loss: {:.4f}'.format(
                epoch, train_loss / len(self.train_loader.dataset)))
        
    def predict(self, x_test):
        prediction, _ = self.model.encode(x_test)
        return prediction.data.numpy()

class BookVectorData(torch.utils.data.Dataset):
    def __init__(self, train_features):
        self.train_features = train_features

    def __getitem__(self, index):
        target = self.train_features[index]
        return target

    def __len__(self):
        return len(self.train_features)
        
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
        # TODO : add the mu, var layer over here.
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