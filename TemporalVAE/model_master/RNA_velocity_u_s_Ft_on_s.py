from . import *
from torch import nn
from . import BaseVAE
import torch
from .types_ import *
# 2023-10-04 22:21:36 change model name from SuperviseVanillaVAE_regressionClfDecoder_mouse_mouse_toyDataset to SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial
class RNA_velocity_u_s_Ft_on_s(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 **kwargs) -> None:
        super(RNA_velocity_u_s_Ft_on_s, self).__init__()

        # modules = []
        # hidden_dims = [in_channels, 50]  # 2024-01-19 12:10:28 change
        # hidden_dims = [in_channels, 50, 50]

        # Build fv(u,s,fts)
        modules = [nn.Sequential(nn.Linear(in_channels, in_channels),
                                 nn.BatchNorm1d(in_channels),
                                 nn.LeakyReLU(), ),
                   nn.Sequential(nn.Linear(in_channels, 50),
                                 nn.BatchNorm1d(50),
                                 nn.LeakyReLU(), ),
                   nn.Sequential(nn.Linear(50, 50),
                                 # nn.BatchNorm1d(in_channels),
                                 # nn.Tanh(),
                                 ),
                   ]
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.BatchNorm1d(h_dim),
        #             nn.Tanh()  # 2024-01-19 12:14:37 change to tanh
        #             # nn.LeakyReLU()
        #         )
        #     )
        #     in_channels = h_dim

        self.fv_decoder = nn.Sequential(*modules)
        self.fts_input = None
        self.fts_decoder = None
        # self.init_weights()
        self.relu_detT = nn.ReLU(inplace=True)

    # def init_weights(self):#2023-07-05 16:56:04
    #     def init_module(module):
    #         if isinstance(module, nn.Linear):
    #             #
    #             nn.init.xavier_uniform_(module.weight)
    #             nn.init.constant_(module.bias, 0.0)
    #         elif isinstance(module, nn.BatchNorm1d):
    #             #
    #             nn.init.constant_(module.weight, 1.0)
    #             nn.init.constant_(module.bias, 0.0)
    #         elif isinstance(module, nn.Conv2d):
    #             #
    #             nn.init.xavier_uniform_(module.weight)
    #             nn.init.constant_(module.bias, 0.0)
    #         elif isinstance(module, nn.BatchNorm2d):
    #             #
    #             nn.init.constant_(module.weight, 1.0)
    #             nn.init.constant_(module.bias, 0.0)
    #
    #     self.fv_decoder.apply(init_module)
    def fv_decode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        spliced_data = input[0]
        unspliced_data = input[1]
        t = self.fts_decoder(self.fts_input(spliced_data))
        concat_input = torch.cat((spliced_data, unspliced_data, t), 1)
        result = self.fv_decoder(concat_input)

        v = torch.flatten(result, start_dim=1)

        return v

    def fts_decode(self, input: Tensor) -> List[Tensor]:
        t = self.fts_decoder(self.fts_input(input))
        return t

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        spliced_data = input[0]
        detT = kwargs["detT"]
        detT = torch.reshape(detT, (len(detT), 1)).float()
        v = self.fv_decode(input)  # input:(spliced, unspliced)
        predict_detT = self.fts_decode(v * detT + spliced_data) - self.fts_decode(spliced_data)
        predict_detT2 = self.relu_detT(predict_detT)

        return [predict_detT2, detT, v]

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
        predict_detT = args[0]
        detT = args[1]

        # New: add the clf loss
        # detT = torch.unsqueeze(detT, dim=1).float()
        # 2024-01-19 16:51:00 loss add ass u&v positive, s&v negative
        loss = F.mse_loss(predict_detT, detT) / detT[0] / detT[0]

        return {'loss': loss}
