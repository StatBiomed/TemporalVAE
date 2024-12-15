
import math
import torch
import torch.nn as nn

class TVAE_base(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dims=[], fit_xscale=True, 
        device='cpu', add_batchnorm=False, fc_activate=torch.nn.ReLU()):
        """
        Variational auto-encoder base model:
        An implementation supporting customized hidden layers via a list.

        For the likelihood, there are three common choices for the variance:
        1) a scalar of variance as hyper-parameter, like PCA
        2) a vector of variance as hyper-parameter, like factor analysis
        3) a matrix of variance as amoritized over z, but it can be unstable
        Here, we choose option 2) via fc_x_logstd() for better stabilization.
        We also use torch.clamp to clip the very small values.
        Now, we also leave this choice as an argument of the loss function,
        including option 0) with a predefined value 1, like MSE loss.

        Parameters
        ----------
        
        Examples
        --------
        my_VAE = VAE_base()
        my_VAE.encoder = resnet18_encoder(False, False) # to change encoder
        """
        super(TVAE_base, self).__init__()
        self.device = device

        # check hiden layers
        # TODO: check int and None
        H = len(hidden_dims)
        encode_dim = x_dim if H == 0 else hidden_dims[-1]
        decode_dim = z_dim if H == 0 else hidden_dims[0]

        # encoder
        self.encoder = torch.nn.Sequential(nn.Identity())
        for h, out_dim in enumerate(hidden_dims):
            in_dim = x_dim if h == 0 else hidden_dims[h - 1]
            self.encoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            if add_batchnorm:
                self.encoder.add_module("N%s" %(h), nn.BatchNorm1d(out_dim)) 
            self.encoder.add_module("A%s" %(h), fc_activate)

        # latent mean and diagonal variance 
        self.fc_z_mean = nn.Linear(encode_dim, z_dim)
        self.fc_z_logstd = nn.Linear(encode_dim, z_dim)
        
        # decoder
        self.decoder = nn.Sequential()
        for h, out_dim in enumerate(hidden_dims[::-1]):
            in_dim = z_dim if h == 0 else hidden_dims[::-1][h - 1]
            self.decoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            if add_batchnorm:
                self.decoder.add_module("N%s" %(h), nn.BatchNorm1d(out_dim))            
            self.decoder.add_module("A%s" %(h), fc_activate)

        # reconstruction mean and diagonal variance (likelihood)
        self.fc_x_mean = nn.Linear(decode_dim, x_dim)
        self.fc_x_logstd = nn.Linear(1, x_dim, bias=False) #(1, 1)
        
        # time predictor
        self.predictor = nn.Sequential(
            nn.Linear(z_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dims[-1], 2),
            nn.BatchNorm1d(2),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.Linear(2, 1))
        
    def encode(self, x):
        """For variational posterior distribution"""
        _x = self.encoder(x)
        z_mean, z_logstd = self.fc_z_mean(_x), self.fc_z_logstd(_x)
        return z_mean, z_logstd

    def reparameterization(self, z_mean, z_logstd):
        epsilon = torch.randn_like(z_mean).to(self.device)
        z = z_mean + torch.exp(z_logstd) * epsilon
        return z

    def decode(self, z):
        """For exact posterior by reconstruction-based likelihood"""
        _z = self.decoder(z)
        x_mean = self.fc_x_mean(_z)
        x_logstd = self.fc_x_logstd(torch.ones(1, 1).to(self.device))
        x_logstd = torch.clamp(x_logstd, min=-2, max=5)
        return x_mean, x_logstd
    
    def predict(self, z):
        # TODO: to consider the variance of the prediction
        y_mean = self.predictor(z)
        return y_mean

    def forward(self, x):
        z_mean, z_logstd = self.encode(x)
        z = self.reparameterization(z_mean, z_logstd)
        x_hat, x_logstd = self.decode(z)
        y_hat = self.predict(z)
        return x_hat, x_logstd, z, z_mean, z_logstd, y_hat
