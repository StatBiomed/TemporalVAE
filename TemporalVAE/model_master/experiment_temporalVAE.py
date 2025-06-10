import os
from torch import optim
from . import BaseVAE
from . import *
import pytorch_lightning as pl
import torchvision.utils as vutils
import numpy as np



class temporalVAEExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(temporalVAEExperiment, self).__init__()
        # self.training_step_outputs = []
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    # def adversarial_loss(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, donor_index = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                              batch_weight=self.params["batch_weight"],  # 2023-09-21 16:53:02
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=True, on_step=True,
                      on_epoch=True, prog_bar=True, logger=True)
        # self.training_step_outputs.append(train_loss)
        return train_loss['loss']

    def on_train_epoch_end(self) -> None:
        # do something with all training_step outputs, for example:
        # epoch_mean = torch.stack(self.training_step_outputs).mean()
        print("Epoch train loss: {}".format(self.trainer.logged_metrics))
        # self.log("training_epoch_loss_not_mean_because_onlyOneEpoch",self.trainer.logged_metrics )
        # free up the memory
        # self.training_step_outputs.clear()

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, donor_index = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        test_loss = self.model.loss_function(*results,
                                             M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                             clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                             batch_weight=self.params["batch_weight"],  # 2023-09-21 16:53:02
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
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        test_loss = self.model.loss_function(*results,
                                             M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                             clf_weight=self.params["clf_weight"],  # 2023-09-21 16:53:02
                                             batch_weight=self.params["batch_weight"],  # 2023-09-21 16:53:02
                                             optimizer_idx=optimizer_idx,
                                             batch_idx=batch_idx)
        if results[0].shape[1] != results[1].shape[1]:  # 2023-06-29 18:27:01 for noCLF decoder vae
            recons=results[0][:,5:].cpu().numpy() #2023-10-15 16:03:40
            reclf = results[0][:, :5].cpu().numpy()
            mu = results[2]
            log_var = results[3]
        else:  # 2023-06-29 18:27:22 for CLF decoder vae
            recons=results[0].cpu().numpy()
            reclf = results[4].cpu().numpy()
            mu = results[2]
            log_var = results[3]
        np.savetxt(self.logger.log_dir + "/prediction_on_test_cell.txt", reclf, delimiter="\t", encoding='utf-8', fmt="%s")
        # self.log_dict({f"predict_{key}": val.item() for key, val in test_loss.items()}, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return reclf, mu, log_var, test_loss,recons

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels, donor_index = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels, batchEffect_real=donor_index)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],  # real_img.shape[0]/ self.num_val_imgs,
                                            clf_weight=self.params["clf_weight"],
                                            batch_weight=self.params["batch_weight"],# 2023-09-21 16:53:02
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


