# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：experiment_RNA_velocity.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024-01-15 15:07:29
"""
from torch import optim
from . import *
# from utils import data_loader
import pytorch_lightning as pl
import numpy as np

BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


class Experiment_RNA_velocity(pl.LightningModule):
    def __init__(self,
                 velocity_model: RNA_velocity_u_s_Ft_on_s,
                 params: dict) -> None:
        super(Experiment_RNA_velocity, self).__init__()
        # self.training_step_outputs = []
        self.model = velocity_model
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

    def train(self, mode=True):
        super(Experiment_RNA_velocity, self).train(mode=mode)
        # only training, change the module's state
        if self.model.training:
            # _make_trainable(module=self.model.encoder)
            # _make_trainable(module=self.model.fc_mu)
            # _make_trainable(module=self.model.fc_var)
            #
            # _make_trainable(module=self.model.decoder_input)
            # _make_trainable(self.model.decoder)
            # _make_trainable(self.model.final_layer)

            freeze(module=self.model.fts_input, train_bn=self.params["train_bn"])
            freeze(module=self.model.fts_decoder, train_bn=self.params["train_bn"])
            # freeze(module=self.model.clf_decoder, train_bn=self.params["train_bn"])

    def training_step(self, batch, batch_idx):
        self.check_each_layer_requires_grad()
        # real_img, detT = batch
        spliced_data, unspliced_data, detT = batch
        real_img = (spliced_data, unspliced_data)
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device
        results = self.forward(real_img, detT=detT)
        train_loss = self.model.loss_function(*results,
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

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, detT = batch
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device
        results = self.forward(real_img, detT=detT)
        test_loss = self.model.loss_function(*results,
                                             optimizer_method="optimize_adam",
                                             # batch_idx=batch_idx
                                             )
        predict_detT = results[0].cpu().numpy()
        detT = results[1].cpu().numpy()
        v = results[3].cpu().numpy()
        # np.savetxt(self.logger.log_dir + "/predict_detT_test_on_test_cell.txt", predict_detT, delimiter="\t", encoding='utf-8', fmt="%s")
        # np.savetxt(self.logger.log_dir + "/detT_test_on_test_cell.txt", predict_detT, delimiter="\t", encoding='utf-8', fmt="%s")
        self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, optimizer_idx=0):

        spliced_data, unspliced_data, detT = batch
        real_img = (spliced_data, unspliced_data)
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device

        results = self.forward(real_img, detT=detT)
        predicte_loss = self.model.loss_function(*results,
                                                 optimizer_method="optimize_adam",
                                                 # batch_idx=batch_idx
                                                 )
        predict_detT = results[0].cpu().numpy()
        detT = results[1].cpu().numpy()
        v = results[2].cpu().numpy()
        np.savetxt(self.logger.log_dir + "/v_prediction_on_test_cell.txt", predict_detT, delimiter="\t", encoding='utf-8', fmt="%s")
        np.savetxt(self.logger.log_dir + "/detT_prediction_on_test_cell.txt", v, delimiter="\t", encoding='utf-8', fmt="%s")
        return predict_detT, v

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, detT = batch
        try:
            self.curr_device = real_img.device
        except:
            self.curr_device = real_img[0].device
        results = self.forward(real_img, detT=detT)
        val_loss = self.model.loss_function(*results,
                                            optimizer_method="optimize_adam",
                                            # batch_idx=batch_idx
                                            )
        self.log_dict({f"test_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True, on_step=True, on_epoch=True,
                      prog_bar=True, logger=True)

    def check_each_layer_requires_grad(self):
        fts_layers_requires_grad_list = [_param.requires_grad for _param in self.model.fts_decoder.parameters()]
        fv_layers_requires_grad_list = [_param.requires_grad for _param in self.model.fv_decoder.parameters()]
        requires_grad_dic = {"fts": fts_layers_requires_grad_list,
                             "fv": fv_layers_requires_grad_list, }
        print("Epoch:{}, each module grad: {}".format(self.current_epoch, requires_grad_dic))

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
