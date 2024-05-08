import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class SuperviseVanillaVAE_removerBatchEffectDecoder(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SuperviseVanillaVAE_removerBatchEffectDecoder, self).__init__()

        self.latent_dim = latent_dim  # latent_dim= batchEffect_dim + for_reconstruate_&_clf_dim
        self.batchEffect_dim = 5
        self.label_num = 5
        self.batch_num = 18  # donor info consider as batch effect
        modules = []
        if hidden_dims is None:
            hidden_dims = [in_channels, 256, 128, 64]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim - self.batchEffect_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # Build classifier, only for discrete label 5, so last layer is sigmoid
        self.clf_input = nn.Linear(self.latent_dim - self.batchEffect_dim, hidden_dims[0])  # hidden_dims[0]=64
        self.clf_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
                                         nn.BatchNorm1d(hidden_dims[0]),
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_dims[0], self.label_num),
                                         nn.BatchNorm1d(self.label_num),
                                         nn.Sigmoid())
        # Build batch effect classifier, also discrete, so last layer is sigmoid
        self.batchEffect_input = nn.Linear(self.batchEffect_dim, self.batchEffect_dim)
        self.batchEffect_decoder = nn.Sequential(nn.Linear(self.batchEffect_dim, self.batch_num),
                                                 nn.BatchNorm1d(self.batch_num),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(self.batch_num, self.batch_num),
                                                 nn.BatchNorm1d(self.batch_num),
                                                 nn.Sigmoid())
        # final_layer for decoder, as the gene expression is N(0,1), the last layer should be tanh
        self.final_layer = nn.Sequential(nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                                         nn.BatchNorm1d(hidden_dims[-1]),
                                         nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        z_sub = z[:, self.batchEffect_dim:]
        result = self.decoder_input(z_sub)  # result shape:(batchSize,64)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def clf_decode(self, z: Tensor) -> Tensor:
        # New: add a classifier
        z_sub = z[:, self.batchEffect_dim:]
        result = self.clf_input(z_sub)  # result shape:(batchSize,64)
        result = self.clf_decoder(result)
        return result

    def batchEffect_decode(self, z: Tensor) -> Tensor:
        # New: add a classifier
        z_sub = z[:, :self.batchEffect_dim]
        result = self.batchEffect_input(z_sub)  # result shape:(batchSize,64)
        result = self.batchEffect_decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        labels = kwargs["labels"]
        batchEffect_real = kwargs["batchEffect_real"]
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        reclf = self.clf_decode(z)
        batchEffect_pred = self.batchEffect_decode(z)
        return [recons, input, mu, log_var, reclf, labels, batchEffect_pred, batchEffect_real]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        reclf = args[4]
        label = args[5]
        batchEffect_pred = args[6]
        batchEffect_label = args[7]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # New: add the clf loss, as label is discrete so cross_entropy
        clf_loss = F.cross_entropy(reclf, label)
        # New: ad the batch effect loss, as the batch label is discrete so cross_entropy
        batchEffect_loss = F.cross_entropy(batchEffect_pred, batchEffect_label.type(torch.cuda.LongTensor))
        loss = recons_loss + kld_weight * kld_loss + clf_loss + batchEffect_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach(), "clf_loss": clf_loss, "batchEffect_loss": batchEffect_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
