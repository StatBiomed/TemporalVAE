# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：experiment_adversarial.py
@IDE     ：PyCharm 
@Author  ：awa121
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


# class ParameterChangeHook:
#     def __init__(self, model):
#         self.handles = []
#         self.model = model
#         self.init_params = [p.clone().detach() for p in model.parameters()]
#
#         for param in model.parameters():
#             handle = param.register_hook(self.param_change_hook)
#             self.handles.append(handle)
#
#     def param_change_hook(self, grad):
#         index = self.handles.index(grad.register_hook[0][1])
#         print(f"Parameter {index} has changed.")
#
#     def remove_hooks(self):
#         for handle in self.handles:
#             handle.remove()


# class CheckBatchGradient(LearningRateMonitor):
#     """Unfreeze feature extractor callback."""
#
#     def on_train_epoch_start(self, trainer, model):
#         n = 0
#
#         example_input = model.example_input_array.to(model.device)
#         example_input.requires_grad = True
#
#         model.zero_grad()
#         output = model(example_input)
#         output[n].abs().sum().backward()
#
#         zero_grad_inds = list(range(example_input.size(0)))
#         zero_grad_inds.pop(n)
#
#         # if example_input.grad[zero_grad_inds].abs().sum().item() > 0
#         #     raise RuntimeError("Your model mixes data across the batch dimension!")
#         # if trainer.current_epoch not in pl_module.params["epoch_milestone"]:
#         #     # model = trainer.get_model()
#         #     # trainer.model.model.encoder
#         #     _make_trainable(trainer.model.model.encoder)
#         #     _make_trainable(trainer.model.model.decoder)
#         #     _make_trainable(trainer.model.model.clf_decoder)
#         #
#         #     # optimizer = trainer.optimizers[0]
#         #     # _current_lr = optimizer.param_groups[0]['lr']
#         #     # optimizer.add_param_group({
#         #     #     'params': filter_params(module=model.features_extractor,
#         #     #                             train_bn=pl_module.hparams.train_bn),
#         #     #     'lr': _current_lr
#         #     # })


class VAEXperiment_adversarial(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment_adversarial, self).__init__()
        # self.training_step_outputs = []
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.automatic_optimization = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def train(self, mode=True):
        super(VAEXperiment_adversarial, self).train(mode=mode)
        # only training, change the module's state
        if self.model.training:
            if self.current_epoch in self.params["epoch_milestone"]:
                print(f"*** epoch is {self.current_epoch}, freeze encoder, decoder, clf_decoder.")
                freeze(module=self.model.encoder, train_bn=self.params["train_bn"])
                freeze(module=self.model.fc_mu, train_bn=self.params["train_bn"])  # 2023-12-06 11:52:12 add
                freeze(module=self.model.fc_var, train_bn=self.params["train_bn"])  # 2023-12-06 11:52:12 add

                freeze(module=self.model.decoder_input, train_bn=self.params["train_bn"])
                freeze(module=self.model.decoder, train_bn=self.params["train_bn"])
                freeze(module=self.model.final_layer, train_bn=self.params["train_bn"])

                freeze(module=self.model.clf_input, train_bn=self.params["train_bn"])
                freeze(module=self.model.clf_decoder, train_bn=self.params["train_bn"])

                _make_trainable(self.model.batchEffect_input)
                _make_trainable(self.model.batchEffect_decoder)


            else:
                print(f"*** epoch is {self.current_epoch}, freeze batchEffect_decoder.")
                _make_trainable(self.model.encoder)
                _make_trainable(self.model.fc_mu)  # 2023-12-06 11:52:12 add
                _make_trainable(self.model.fc_var)  # 2023-12-06 11:52:12 add

                _make_trainable(self.model.decoder_input)
                _make_trainable(self.model.decoder)
                _make_trainable(self.model.final_layer)

                _make_trainable(self.model.clf_decoder)
                _make_trainable(self.model.clf_input)

                freeze(module=self.model.batchEffect_decoder, train_bn=self.params["train_bn"])
                freeze(module=self.model.batchEffect_input, train_bn=self.params["train_bn"])

    def training_step(self, batch, batch_idx):
        self.check_each_layer_requires_grad()

        real_img, labels, donor_index = batch
        self.curr_device = real_img.device

        optimizer_en, optimizer_batch = self.optimizers()

        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)

        if self.current_epoch in self.params["epoch_milestone"]:
            # Step 1: with the encoder fixed in step 0, now train another classifier to predict batch, then we fix this classifier
            # self.toggle_optimizer(optimizer_batch)
            train_loss = self.model.loss_function(*results,
                                                  M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  clf_weight=self.params["clf_weight"],  # 2023-09-13 13:50:42
                                                  batch_weight=self.params["batch_weight"],  # 2023-08-14 15:49:40
                                                  optimizer_method="optimizer_batch",
                                                  batch_idx=batch_idx,
                                                  adversarial_bool=False)
            optimizer_batch.zero_grad()
            self.manual_backward(train_loss['loss'])
            optimizer_batch.step()
            # self.untoggle_optimizer(optimizer_batch)
        elif self.current_epoch in self.params["epoch_adversarial"]:
            # self.toggle_optimizer(optimizer_en)
            # Step 2: retrain the VAE as step 0, but this time we add one more loss to make batch classifier in step 1 as bad as possible.
            train_loss = self.model.loss_function(*results,
                                                  M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  clf_weight=self.params["clf_weight"],  # 2023-09-13 13:50:42
                                                  batch_weight=self.params["batch_weight"],  # 2023-08-14 15:49:40
                                                  optimizer_method="optimizer_en",
                                                  batch_idx=batch_idx,
                                                  adversarial_bool=True)
            optimizer_en.zero_grad()
            self.manual_backward(train_loss['loss'])
            optimizer_en.step()
            # self.untoggle_optimizer(optimizer_en)
        else:
            # Step 0: train our VAE with two decoders (reconstruction & time) - this is what we have for now
            # self.toggle_optimizer(optimizer_en)
            train_loss = self.model.loss_function(*results,
                                                  M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                  clf_weight=self.params["clf_weight"],  # 2023-09-13 13:50:42
                                                  batch_weight=self.params["batch_weight"],  # 2023-08-14 15:49:40
                                                  optimizer_method="optimizer_en",
                                                  batch_idx=batch_idx,
                                                  adversarial_bool=False)
            optimizer_en.zero_grad()
            self.manual_backward(train_loss['loss'])
            optimizer_en.step()
            # self.untoggle_optimizer(optimizer_en)

        if self.trainer.is_last_batch:
            # log grad in the last batch of each epoch
            for name, param in self.model.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.current_epoch)
                if param.requires_grad:
                    try:
                        self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.current_epoch)
                    except:
                        print(name, "error?")
            # multiple schedulers
            sch_en, sch_batch = self.lr_schedulers()
            if self.current_epoch in self.params["epoch_milestone"]:
                sch_batch.step()
            else:
                sch_en.step()
        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True, on_step=True,
                      on_epoch=True, prog_bar=True, logger=True)

        # return train_loss['loss']

    def on_train_epoch_end(self) -> None:
        #  the function is called after every epoch is completed
        # do something with all training_step outputs, for example:
        # epoch_mean = torch.stack(self.training_step_outputs).mean()

        print("Epoch train loss: {}".format(self.trainer.logged_metrics))
        optimizer_en, optimizer_batch = self.optimizers()
        print(f"en lr:{optimizer_en.optimizer.param_groups[0]['lr']}")
        print(f"batch lr: {optimizer_batch.optimizer.param_groups[0]['lr']}")
        # free up the memory
        if (self.current_epoch + 1) % 20 == 0:
            torch.cuda.empty_cache()
        self.train()


    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            real_img, labels, donor_index = batch
            self.curr_device = real_img.device

            results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
            test_loss = self.model.loss_function(*results,
                                                 M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                 clf_weight=self.params["clf_weight"],  # 2023-09-13 13:50:42
                                                 batch_weight=self.params["batch_weight"],  # 2023-08-14 15:49:40
                                                 optimizer_method="optimizer_en",
                                                 batch_idx=batch_idx, adversarial_bool=False)

            if results[0].shape[1] != results[1].shape[1]:  # 2023-06-29 18:27:01 for noCLF decoder vae
                reclf = results[0][:, :5].cpu().numpy()
            else:  # 2023-06-29 18:27:22 for CLF decoder vae
                reclf = results[4].cpu().numpy()
            np.savetxt(self.logger.log_dir + "/test_on_test_cell.txt", reclf, delimiter="\t", encoding='utf-8', fmt="%s")
            self.log_dict({f"test_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True, on_step=True,
                          on_epoch=True,
                          prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            real_img, labels, donor_index = batch
            self.curr_device = real_img.device

            results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
            test_loss = self.model.loss_function(*results,
                                                 M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                 clf_weight=self.params["clf_weight"],  # 2023-09-13 13:50:42
                                                 batch_weight=self.params["batch_weight"],  # 2023-08-14 15:49:40
                                                 optimizer_method="optimizer_en",
                                                 batch_idx=batch_idx, adversarial_bool=False)

            if results[0].shape[1] != results[1].shape[1]:  # 2023-06-29 18:27:01 for noCLF decoder vae
                recons = results[0][:, 5:].cpu().numpy()  # 2023-10-15 16:03:40
                reclf = results[0][:, :5].cpu().numpy()
                mu = results[2]
                log_var = results[3]
            else:  # 2023-06-29 18:27:22 for CLF decoder vae
                recons = results[0].cpu().numpy()
                reclf = results[4].cpu().numpy()
                mu = results[2]
                log_var = results[3]
            np.savetxt(self.logger.log_dir + "/prediction_on_test_cell.txt", reclf, delimiter="\t", encoding='utf-8', fmt="%s")
            # self.log_dict({f"predict_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return reclf, mu, log_var, test_loss, recons

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            real_img, labels, donor_index = batch
            self.curr_device = real_img.device
            results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)

            test_loss = self.model.loss_function(*results,
                                                 M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                                 clf_weight=self.params["clf_weight"],  # 2023-09-13 13:50:42
                                                 batch_weight=self.params["batch_weight"],  # 2023-08-14 15:49:40
                                                 optimizer_method="optimizer_en",
                                                 batch_idx=batch_idx, adversarial_bool=False)

            self.log_dict({f"val_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True, on_epoch=True,
                          logger=True)

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
        batch_decoder_layers_requires_grad_list = [_param.requires_grad for _param in self.model.batchEffect_decoder.parameters()]
        requires_grad_dic = {"encoder": encoder_layers_requires_grad_list,
                             "decoder": decoder_layers_requires_grad_list,
                             "clf_decoder": clf_decoder_layers_requires_grad_list,
                             "batch_decoder": batch_decoder_layers_requires_grad_list}
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
                # 2023-08-15 17:53:10
                # scheduler = optim.lr_scheduler.CyclicLR(optims[0], cycle_momentum=False,
                #                                         max_lr=self.params['LR'],
                #                                         base_lr=1e-4,
                #                                         step_size_up=20,
                #                                         gamma=self.params['scheduler_gamma'])
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        # scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                        #                                               gamma=self.params['scheduler_gamma_2'])
                        # scheduler2 = optim.lr_scheduler.CyclicLR(optims[1], cycle_momentum=False,
                        #                                         max_lr=0.001,
                        #                                         base_lr=1e-4,
                        #                                         step_size_up=10,
                        #                                         gamma=self.params['scheduler_gamma_2'])
                        def lr_lambda(epoch):
                            # 每20个epoch重置一次周期
                            cycle_length = 10
                            cycle_epoch = epoch % cycle_length

                            # 在20个epoch内从0.001降到0.0001
                            # start_lr = 0.001
                            # end_lr = 0.0001
                            start_lr = 0.01
                            end_lr = 0.0001
                            # start_lr = 0.0001
                            # end_lr = 0.01
                            lr_decay = (end_lr / start_lr) ** (1 / (cycle_length - 1))  # 计算每个epoch的衰减率
                            return lr_decay ** cycle_epoch  # 返回乘以初始学习率的因子
                        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optims[1], lr_lambda)
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
