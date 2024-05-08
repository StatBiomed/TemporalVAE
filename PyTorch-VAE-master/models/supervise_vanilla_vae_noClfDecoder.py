import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class SuperviseVanillaVAE_noCLFDecoder(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SuperviseVanillaVAE_noCLFDecoder, self).__init__()

        self.latent_dim = latent_dim

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
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        # 2023-06-28 20:05:12 only change decoder, no clf as a decoder
        hidden_dims[-1] = hidden_dims[-1] + 5  # 2023-06-29 18:01:29 5 is the number of label type
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        # 2023-06-29 18:01:29 5 is the number of label type, new
        self.final_layer_for_recon = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] - 5),
            nn.BatchNorm1d(hidden_dims[-1] - 5),
            nn.LeakyReLU(),
            nn.Tanh())
        self.final_layer_for_reclf = nn.Sequential(
            nn.Linear(hidden_dims[-1], 5),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(),
            nn.Softmax())

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
        result = self.decoder_input(z)  # result shape:(batchSize,64)
        result = self.decoder(result)
        reclf = self.final_layer_for_reclf(result)
        recons = self.final_layer_for_recon(result)
        # 按列合并两个 Tensor
        reclf_recons = torch.cat((reclf, recons), dim=1)
        return reclf_recons

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
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        # reclf = self.clf_decode(z)
        return [recons, input, mu, log_var, labels]

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
        reclf_recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        label = args[4]

        reclf = reclf_recons[:, :5]
        recons = reclf_recons[:, 5:]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        clf_loss = F.cross_entropy(reclf, label)

        # # 2023-06-29 18:05:52 对每一行进行归一化到 0 到 1 之间的操作
        # normalized_tensor = F.normalize(reclf, p=1, dim=1)
        #
        # # 使用 softmax 函数对每一行进行处理
        # softmax_tensor = F.softmax(normalized_tensor, dim=1)

        loss = recons_loss + kld_weight * kld_loss + clf_loss

        # calculate acc
        labels_pred = torch.argmax(reclf, dim=1)
        acc = torch.tensor(torch.sum(labels_pred == label).item() / (len(label) * 1.0))
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach(), "clf_loss": clf_loss.detach(), "acc": acc}

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
