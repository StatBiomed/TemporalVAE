import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


# 2023-10-04 22:21:36 change model name from SuperviseVanillaVAE_regressionClfDecoder_mouse_mouse_toyDataset to SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial
class SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusDecoder(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_u_s_focusDecoder, self).__init__()

        self.latent_dim = latent_dim
        if "label_num" in kwargs:
            self.label_num = kwargs["label_num"]
        else:
            self.label_num = 10
        modules = []
        if hidden_dims is None:
            hidden_dims = [in_channels, 256, 128, 64]

        # Build Encoder
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.BatchNorm1d(in_channels),
                nn.Tanh()
            ))
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        # Build classifier, only for continue, so last layer use tanh
        self.clf_input = nn.Linear(self.latent_dim, hidden_dims[0])
        self.clf_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
                                         nn.BatchNorm1d(hidden_dims[0]),
                                         nn.LeakyReLU(),
                                         nn.Dropout(p=0.5),  # 2023-09-21 17:06:34 add
                                         nn.Linear(hidden_dims[0], self.label_num * 2),
                                         nn.BatchNorm1d(self.label_num * 2),
                                         nn.Tanh(),
                                         nn.Dropout(p=0.5),  # 2023-09-21 17:06:34 add
                                         nn.Linear(self.label_num * 2, self.label_num * 2),  # 2023-09-21 17:06:34 add
                                         nn.BatchNorm1d(self.label_num * 2),  # 2023-09-21 17:06:34 add
                                         nn.Tanh(),  # 2023-09-21 17:06:34 add
                                         # nn.Dropout(p=0.5),  # 2023-09-21 17:06:34 add
                                         nn.Linear(self.label_num * 2, 1))

        # final_layer for decoder, as the gene expression is N(0,1), the last layer should be tanh
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # spliced_data = input[0]
        # unspliced_data = input[1]

        # predicted_encoder_unspliced= self.encoder._modules["0"]._modules["splice_layer"](spliced_data)

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
        decoder_output = self.decoder_input(z)  # result shape:(batchSize,64)
        # result = self.decoder(result)
        unspliced_output = self.decoder(decoder_output)
        unspliced_to_spliced = self.final_layer(unspliced_output)
        spliced_output = unspliced_output + unspliced_to_spliced

        return spliced_output, unspliced_output

    def clf_decode(self, z: Tensor) -> Tensor:
        # New: add a classifier
        result = self.clf_input(z)  # result shape:(batchSize,64)
        result = self.clf_decoder(result)
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
        spliced_data = input[0]
        # unspliced_data = input[1]
        mu, log_var = self.encode(spliced_data)  # input:(spliced, unspliced)
        z = self.reparameterize(mu, log_var)
        spliced_recons, unspliced_recons = self.decode(z)
        reclf = self.clf_decode(z)
        return [(spliced_recons, unspliced_recons), input, mu, log_var, reclf, labels]

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
        spliced_recons, unspliced_recons = args[0]
        spliced_input, unspliced_input = args[1]
        mu = args[2]
        log_var = args[3]
        reclf = args[4]
        label = args[5]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        clf_weight = kwargs['clf_weight']  # Account for the minibatch samples from the dataset

        spliced_recons_loss = F.mse_loss(spliced_recons, spliced_input)
        unspliced_recons_loss = F.mse_loss(unspliced_recons, unspliced_input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # New: add the clf loss
        label_reshape = torch.unsqueeze(label, dim=1).float()
        clf_loss = F.mse_loss(reclf, label_reshape)
        # clf_loss = clf_weight * F.cross_entropy(reclf, label)

        loss = spliced_recons_loss + unspliced_recons_loss + kld_weight * kld_loss + clf_weight * clf_loss
        return {'loss': loss,
                'spliced_reconstruction_loss': spliced_recons_loss.detach(),
                'unspliced_reconstruction_loss': unspliced_recons_loss.detach(),
                'KLD': -kld_loss.detach(), "clf_loss": clf_loss}

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