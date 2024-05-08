import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class SuperviseVanillaVAE_regressionClfDecoder_adversarial(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 batch_num: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SuperviseVanillaVAE_regressionClfDecoder_adversarial, self).__init__()

        self.latent_dim = latent_dim
        self.batch_num = batch_num  # 2023-08-11 11:18:04 donor info consider as batch effect
        if "label_num" in kwargs:
            self.label_num = kwargs["label_num"]
        else:
            self.label_num = 5
        if "fl_gamma" in kwargs:
            self.fl_gamma = kwargs["fl_gamma"]
        else:
            self.fl_gamma = 2

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
        # self.clf_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
        #                                  nn.BatchNorm1d(hidden_dims[0]),
        #                                  nn.LeakyReLU(),
        #                                  nn.Linear(hidden_dims[0], self.label_num * 2),
        #                                  nn.BatchNorm1d(self.label_num * 2),
        #                                  nn.Tanh(),
        #                                  nn.Linear(self.label_num * 2, 1))
        # 2023-10-04 22:13:56 change clf_decoder structure
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

        # 2023-08-11 11:21:43 Build batch effect classifier, also discrete, so last layer is sigmoid
        self.batchEffect_input = nn.Linear(self.latent_dim, hidden_dims[0])
        # self.batchEffect_decoder = nn.Sequential(nn.BatchNorm1d(hidden_dims[0]),
        #                                          nn.ReLU(),
        #                                          nn.Linear(hidden_dims[0], hidden_dims[0]),
        #                                          nn.BatchNorm1d(hidden_dims[0]),
        #                                          nn.ReLU(),
        #                                          nn.Linear(hidden_dims[0], self.batch_num),
        #                                          nn.Sigmoid())
        '''
        before 2024-04-07 10:46:55 version
        self.batchEffect_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
                                                 nn.BatchNorm1d(hidden_dims[0]),
                                                 nn.LeakyReLU(),
                                                 nn.Dropout(p=0.5),  # 2023-09-12 20:03:01
                                                 nn.Linear(hidden_dims[0], hidden_dims[0]),
                                                 nn.BatchNorm1d(hidden_dims[0]),
                                                 nn.LeakyReLU(),
                                                 nn.Dropout(p=0.5),  # 2023-09-12 20:03:01
                                                 nn.Linear(hidden_dims[0], self.batch_num),
                                                 nn.Linear(self.batch_num, self.batch_num),  
                                                 nn.Sigmoid())
        '''
        # 2024-04-07 10:47:34 change for human embryo dataset
        self.batchEffect_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
                                                 nn.BatchNorm1d(hidden_dims[0]),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(hidden_dims[0], 20),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(20, 10),
                                                 nn.LeakyReLU(),
                                                 nn.Linear(10, self.batch_num),
                                                 # nn.Sigmoid()
                                                 )
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
        result = self.final_layer(result)
        return result

    def clf_decode(self, z: Tensor) -> Tensor:
        # New: add a classifier
        result = self.clf_input(z)  # result shape:(batchSize,64)
        result = self.clf_decoder(result)
        return result

    def batchEffect_decode(self, z: Tensor) -> Tensor:
        # New: add a classifier
        result = self.batchEffect_input(z)  # result shape:(batchSize,64)
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
        rebatchEffect = self.batchEffect_decode(z)

        return [recons, input, mu, log_var, reclf, labels, rebatchEffect, batchEffect_real]

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
        labels = args[5]
        batch_pred = args[6]
        batch_label = args[7]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        clf_weight = kwargs['clf_weight']  # Account for the minibatch samples from the dataset
        batch_weight = kwargs["batch_weight"]  # 2023-08-14 15:50:52
        optimizer_method = kwargs['optimizer_method']
        # Update encoder, decoder, clf-decoder  # how well can it label as real?
        # batch_loss = batch_weight * FocalLoss(gamma=self.fl_gamma)(batch_pred, batch_label.type(torch.cuda.LongTensor))
        # 选取属于类别1的概率
        # probabilities = batch_pred[:, 1]
        # batch_loss = batch_weight * F.binary_cross_entropy(probabilities, batch_label)

        # max_indices = torch.argmax(batch_pred, dim=1)
        # F.binary_cross_entropy(max_indices, batch_label.type(torch.cuda.IntTensor))
        # F.cross_entropy(batch_pred, batch_label.type(torch.cuda.LongTensor))

        # batch_loss = batch_weight * F.cosine_similarity(batch_pred, batch_label.type(torch.cuda.LongTensor))
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # New: add the clf loss
        label_reshape = torch.unsqueeze(labels, dim=1).float()

        # 2024-03-20 20:34:21 the normal one, need to change after these exps
        clf_loss = clf_weight * F.mse_loss(reclf, label_reshape)

        # 2024-03-20 20:27:11 from Huang, only calculate atlas data clf loss
        # reversed_batch_label = 1 - batch_label
        # 将reversed_batch_label从[0, 1]布尔掩码转换为与reclf和label_reshape相同形状的掩码
        # mask = reversed_batch_label.unsqueeze(1).float()
        # 计算所有样本的损失，但只保留mask中为1的样本损失，其他设为0
        # losses = F.mse_loss(reclf, label_reshape, reduction='none') * mask
        # 计算平均损失，但仅包括mask为1的样本
        # clf_loss =clf_weight*( losses.sum() / mask.sum())

        print(f"clf_weight: {clf_weight}")
        if optimizer_method == "optimizer_en":
            # loss: Step 0: train our VAE with two decoders (reconstruction & time) - this is what we have for now
            batch_loss = batch_weight * F.cross_entropy(batch_pred, batch_label.type(torch.cuda.LongTensor))
            loss = recons_loss + kld_weight * kld_loss + clf_loss
            if kwargs["adversarial_bool"]:
                def custom_loss(outputs, targets):
                    """
                    自定义损失函数。
                    outputs: 模型的输出，假设已经经过sigmoid激活，表示正类的概率。
                    targets: 真实的标签，0或1。
                    """
                    outputs = torch.sigmoid(outputs)
                    # 对于不是完全正确或完全错误的情况，计算confidence与0.5的距离
                    confidence_loss = 1 - (outputs - 0.5).abs() * 2  # 将confidence的范围[0, 0.5]映射到[1, 0]
                    return confidence_loss.mean()  # 返回平均损失

                batch_loss = batch_weight * F.cross_entropy(batch_pred, batch_label.type(torch.cuda.LongTensor))
                # batch_loss = batch_weight * custom_loss(batch_pred, batch_label.type(torch.cuda.LongTensor))
                # loss: Step 2: retrain the VAE as step 0, but this time we add one more loss to make batch classifier in step 1 as bad as possible.
                loss = loss - batch_loss
            # return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach(),
            #         "clf_loss": clf_loss, "batchEffect_loss": batch_loss}
        # Update batchEffect-decoder
        elif optimizer_method == "optimizer_batch":
            batch_loss = batch_weight * F.cross_entropy(batch_pred, batch_label.type(torch.cuda.LongTensor))
            # loss: Step 1: with the encoder fixed in step 0, now train another classifier to predict batch, then we fix this classifier
            loss = batch_loss
            # return {'loss': loss, "batchEffect_loss": batch_loss}
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach(),
                "clf_loss": clf_loss, "batchEffect_loss": batch_loss}

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
