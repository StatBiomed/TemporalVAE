# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：experiment_fineTune.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/11/30 13:16 
@Date    ：2023/8/14 11:05
"""
import os
from torch import optim
from . import BaseVAE
from . import *
# from utils import data_loader
import pytorch_lightning as pl
import torchvision.utils as vutils
import numpy as np

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


class VAEXperiment_fineTune_u_s_focusEncoder(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment_fineTune_u_s_focusEncoder, self).__init__()
        # self.training_step_outputs = []
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        # self.automatic_optimization = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def forward_predict(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def train(self, mode=True):
        super(VAEXperiment_fineTune_u_s_focusEncoder, self).train(mode=mode)
        # only training, change the module's state
        if self.model.training:
            _make_trainable(module=self.model.encoder)
            _make_trainable(module=self.model.fc_mu)
            _make_trainable(module=self.model.fc_var)

            _make_trainable(module=self.model.decoder_input)
            _make_trainable(self.model.decoder)
            _make_trainable(self.model.final_layer)

            _make_trainable(self.model.q_u_s_encoder)

            freeze(module=self.model.clf_input, train_bn=self.params["train_bn"])
            freeze(module=self.model.clf_decoder, train_bn=self.params["train_bn"])

    def training_step(self, batch, batch_idx):
        self.check_each_layer_requires_grad()
        real_img, labels, donor_index = batch
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device
        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                              optimizer_method="optimize_adam",
                                              # batch_idx=batch_idx
                                              )
        # self.optimizers().optimizer.zero_grad()
        # self.manual_backward(train_loss['loss'])
        # self.optimizers().optimizer.step()

        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True, on_step=False,
                      on_epoch=True, prog_bar=True, logger=True)

        # self.training_step_outputs.append(train_loss)
        return train_loss['loss']

    def on_train_epoch_end(self) -> None:
        # do something with all training_step outputs, for example:
        # epoch_mean = torch.stack(self.training_step_outputs).mean()
        print("Epoch train loss: {}".format(self.trainer.logged_metrics))
        print(f"lr: {self.optimizers().optimizer.param_groups[0]['lr']}")
        # self.log("training_epoch_loss_not_mean_because_onlyOneEpoch",self.trainer.logged_metrics )
        # free up the memory
        # self.training_step_outputs.clear()
        self.train()

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, donor_index = batch
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device
        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        test_loss = self.model.loss_function(*results,
                                             M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                             clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                             optimizer_idx=optimizer_idx,
                                             batch_idx=batch_idx)
        if results[0].shape[1] != results[1].shape[1]:  # 2023-06-29 18:27:01 for noCLF decoder vae
            reclf = results[0][:, :5].cpu().numpy()
        else:  # 2023-06-29 18:27:22 for CLF decoder vae
            reclf = results[4].cpu().numpy()
        np.savetxt(self.logger.log_dir + "/test_on_test_cell.txt", reclf, delimiter="\t", encoding='utf-8', fmt="%s")
        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, optimizer_idx=0):

        real_img, labels, donor_index = batch
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device

        # def predict_spliced_unspliced(spliced_unspliced_type):
        #     results = self.forward_predict(real_img, labels=labels, batchEffect_real=donor_index,spliced_unspliced_type="spliced_unspliced_type")
        #     test_loss = self.model.loss_function(*results,
        #                                          M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
        #                                          clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
        #                                          optimizer_idx=optimizer_idx,
        #                                          batch_idx=batch_idx)
        #
        #     spliced_recons,unspliced_recons = results[0][0].cpu().numpy(),results[0][1].cpu().numpy()
        #     reclf = results[4].cpu().numpy()
        #     mu = results[2]
        #     log_var = results[3]
        #     np.savetxt(self.logger.log_dir + "/prediction_on_test_cell.txt", reclf, delimiter="\t", encoding='utf-8', fmt="%s")
        #     return (reclf, mu, log_var, test_loss, (spliced_recons,unspliced_recons))
        #
        # return predict_spliced_unspliced("spliced"), predict_spliced_unspliced("unspliced")
        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        test_loss = self.model.loss_function(*results,
                                             M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                             clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                             optimizer_idx=optimizer_idx,
                                             batch_idx=batch_idx)
        # [(spliced_recons, unspliced_recons), input, (s_mu, u_mu), (s_log_var, u_log_var), (s_reclf, u_reclf), labels]
        spliced_recons, unspliced_recons = results[0][0].cpu().numpy(), results[0][1].cpu().numpy()
        s_mu, u_mu = results[2]
        s_log_var, u_log_var = results[3]
        s_reclf, u_reclf = results[4][0].cpu().numpy(), results[4][1].cpu().numpy()

        np.savetxt(self.logger.log_dir + "/splicded_prediction_on_test_cell.txt", s_reclf, delimiter="\t", encoding='utf-8', fmt="%s")
        np.savetxt(self.logger.log_dir + "/unsplicded_prediction_on_test_cell.txt", u_reclf, delimiter="\t", encoding='utf-8', fmt="%s")
        return (s_reclf, s_mu, s_log_var, test_loss, (spliced_recons, unspliced_recons)), (u_reclf, u_mu, u_log_var, test_loss, (spliced_recons, unspliced_recons))

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, donor_index = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],  # real_img.shape[0]/ self.num_val_imgs,
                                            clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True, on_epoch=True, logger=True)

    # def on_validation_end(self) -> None:
    #     # self.sample_images()
    #     print()
    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def check_each_layer_requires_grad(self):
        encoder_layers_requires_grad_list = [_param.requires_grad for _param in self.model.encoder.parameters()]
        decoder_layers_requires_grad_list = [_param.requires_grad for _param in self.model.decoder.parameters()]
        clf_decoder_layers_requires_grad_list = [_param.requires_grad for _param in self.model.clf_decoder.parameters()]
        fc_mu_layers_requires_grad_list = [_param.requires_grad for _param in self.model.fc_mu.parameters()]
        fc_var_layers_requires_grad_list = [_param.requires_grad for _param in self.model.fc_var.parameters()]
        decoder_input_layers_requires_grad_list = [_param.requires_grad for _param in self.model.decoder_input.parameters()]
        clf_input_layers_requires_grad_list = [_param.requires_grad for _param in self.model.clf_input.parameters()]
        final_layers_requires_grad_list = [_param.requires_grad for _param in self.model.final_layer.parameters()]
        q_u_s_encoder_layers_requires_grad_list = [_param.requires_grad for _param in self.model.q_u_s_encoder.parameters()]
        requires_grad_dic = {"encoder": encoder_layers_requires_grad_list,
                             "decoder": decoder_layers_requires_grad_list,
                             "fc_mu": fc_mu_layers_requires_grad_list,
                             "fc_var": fc_var_layers_requires_grad_list,
                             "decoder_input": decoder_input_layers_requires_grad_list,
                             "final_layer": final_layers_requires_grad_list,
                             "q_u_s_encoder": q_u_s_encoder_layers_requires_grad_list,
                             "clf_input": clf_input_layers_requires_grad_list,
                             "clf_decoder": clf_decoder_layers_requires_grad_list, }
        print(f"TRUE means training; FALSE means Freezinng. \n Epoch:{self.current_epoch}, each module grad: {requires_grad_dic}")

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


def _make_trainable(module):
    """Unfreeze a given module.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn=True):
    """Freeze the layers of a given module.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
        Otherwise, they will be set to eval mode along with the other modules.
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module, n=-1, train_bn=True):
    """Freeze the layers up to index n.

    Operates in-place.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    n : int
        By default, all the layers will be frozen. Otherwise, an integer
        between 0 and `len(module.children())` must be given.

    train_bn : bool (default: True)
        If True, the BatchNorm layers will remain in training mode.
    """
    idx = 0
    children = list(module.children())
    if children == [] and (module is not None):
        children = [module]
    n_max = len(children) if n == -1 else int(n)
    for child in children:
        if idx < n_max:
            _recursive_freeze(module=child, train_bn=train_bn)
        else:
            _make_trainable(module=child)


def filter_params(module, train_bn=True):
    """Yield the trainable parameters of a given module.

    Parameters
    ----------
    module : instance of `torch.nn.Module`

    train_bn : bool (default: True)

    Returns
    -------
    generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            filter_params(module=child, train_bn=train_bn)
