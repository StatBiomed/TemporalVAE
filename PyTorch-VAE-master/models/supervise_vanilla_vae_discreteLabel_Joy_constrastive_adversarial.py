import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class SuperviseVanillaVAE_discreteLabel_Joy_constrastive_adversarial(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 batch_num: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(SuperviseVanillaVAE_discreteLabel_Joy_constrastive_adversarial, self).__init__()

        self.latent_dim = latent_dim
        self.batch_num = batch_num  # 2023-08-11 11:18:04 donor info consider as batch effect
        if "label_num" in kwargs:
            self.label_num = kwargs["label_num"]
        else:
            self.label_num = 5  # cell type number
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
        self.clf_input = nn.Linear(self.latent_dim, self.latent_dim)
        # self.clf_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
        #                                  nn.BatchNorm1d(hidden_dims[0]),
        #                                  nn.LeakyReLU(),
        #                                  nn.Linear(hidden_dims[0], self.label_num * 2),
        #                                  nn.BatchNorm1d(self.label_num * 2),
        #                                  nn.Tanh(),
        #                                  nn.Linear(self.label_num * 2, 1))
        # 2023-10-04 22:13:56 change clf_decoder structure
        self.clf_decoder = nn.Sequential(
            # nn.Linear(hidden_dims[0], hidden_dims[0]),
                                         nn.BatchNorm1d(self.latent_dim),
                                         nn.LeakyReLU(),
                                         # nn.Dropout(p=0.5),  # 2023-09-21 17:06:34 add
                                         # nn.Linear(hidden_dims[0], self.label_num * 2),
                                         # nn.BatchNorm1d(self.label_num * 2),
                                         # nn.Tanh(),
                                         # nn.Dropout(p=0.5),  # 2023-09-21 17:06:34 add
                                         # nn.Linear(self.label_num * 2, self.label_num * 2),  # 2023-09-21 17:06:34 add
                                         # nn.BatchNorm1d(self.label_num * 2),  # 2023-09-21 17:06:34 add
                                         # nn.Tanh(),  # 2023-09-21 17:06:34 add
                                         # nn.Linear(self.label_num * 2, self.label_num)
                                         nn.Linear(self.latent_dim, self.latent_dim)
                                         )

        # 2023-08-11 11:21:43 Build batch effect classifier, also discrete, so last layer is sigmoid
        self.batchEffect_input = nn.Linear(self.latent_dim, hidden_dims[0])
        self.batchEffect_decoder = nn.Sequential(nn.Linear(hidden_dims[0], hidden_dims[0]),
                                                 nn.BatchNorm1d(hidden_dims[0]),
                                                 nn.LeakyReLU(),
                                                 nn.Dropout(p=0.5),  # 2023-09-12 20:03:01
                                                 nn.Linear(hidden_dims[0], self.batch_num),
                                                 # nn.Linear(self.batch_num, self.batch_num), # 2023-10-06 15:49:29
                                                 nn.Sigmoid())
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
        # 2023-11-07 15:10:44 use contrastive learning
        result = self.clf_input(z)  # result shape:(batchSize,64)
        # result = self.clf_decoder(result)
        # k-means cluster result

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
        batch_loss = batch_weight * F.cross_entropy(batch_pred, batch_label.type(torch.cuda.LongTensor))
        # max_indices = torch.argmax(batch_pred, dim=1)
        # F.binary_cross_entropy(max_indices, batch_label.type(torch.cuda.IntTensor))
        # F.cross_entropy(batch_pred, batch_label.type(torch.cuda.LongTensor))

        # batch_loss = batch_weight * F.cosine_similarity(batch_pred, batch_label.type(torch.cuda.LongTensor))
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # New: add the clf loss
        # label_reshape = torch.unsqueeze(labels, dim=1).float()

        def kmeans_probabilities(X, num_clusters, iterations):
            # 1. 随机初始化质心
            centroids = X[torch.randperm(X.size(0))][:num_clusters]
            # 2. 迭代优化
            for _ in range(iterations):
                # 2.1 分配到最近的质心
                distances = torch.cdist(X, centroids)
                labels = torch.argmin(distances, dim=1)

                # 2.2 计算新的质心
                centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])

            # 计算“概率”
            distances = torch.cdist(X, centroids)
            probabilities = F.softmax(-distances, dim=1)  # 使用负距离进行softmax，以得到概率
            # probabilities = F.softmax(-distances, dim=1)  # 使用负距离进行softmax，以得到概率

            return labels, centroids, probabilities

        # 指定聚类数和迭代次数
        num_clusters = self.label_num
        iterations = 200

        # 执行 K-means 算法
        cluster_labels, cluster_centroids, cluster_labels_prob = kmeans_probabilities(reclf, num_clusters, iterations)

        # print(cluster_labels)  # 每个点的簇标签
        # print(cluster_centroids)  # 簇质心

        clf_loss = clf_weight * F.cross_entropy(cluster_labels_prob, labels.type(torch.cuda.LongTensor))  # 2023-11-07 15:48:27 change here
        if optimizer_method == "optimizer_en":
            # loss: Step 0: train our VAE with two decoders (reconstruction & time) - this is what we have for now
            loss = recons_loss + kld_weight * kld_loss + clf_loss
            if kwargs["adversarial_bool"]:
                # loss: Step 2: retrain the VAE as step 0, but this time we add one more loss to make batch classifier in step 1 as bad as possible.
                loss = loss - batch_loss
            # return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach(),
            #         "clf_loss": clf_loss, "batchEffect_loss": batch_loss}
        # Update batchEffect-decoder
        elif optimizer_method == "optimizer_batch":
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
