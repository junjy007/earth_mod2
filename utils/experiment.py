from utils.common import *
from models.vae import ChlVAE
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from config.config import Config
import pytorch_lightning as pl
from data.datasets import prepare_data, ChlDataset
import os


class CholDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.data_dir = cfg.data_dir
        if cfg.does_use_small_data:
            # self.fname = cfg.small_data_name # this is for TINY
            self.fname = cfg.data_name
            self.is_small = True
        else:
            self.fname = cfg.data_name
            self.is_small = False

        self.full_path = os.path.join(self.data_dir, self.fname)
        self.batch_size = cfg.batch_size
        self.train_val_split_ratio = cfg.train_val_split_ratio
        self.num_trn_workers = cfg.num_workers_train_loader
        self.num_val_workers = cfg.num_workers_val_loader

        self.data_train, self.data_val = None, None

    def prepare_data(self):
        prepare_data(self.data_dir, self.full_path)

    def setup(self, stage=None):
        kwargs = {}
        if self.is_small:
            kwargs['is_small'] = True
        data_full = ChlDataset(self.full_path, **kwargs)
        n_trn = int(self.train_val_split_ratio * len(data_full))
        n_val = len(data_full) - n_trn
        self.data_train, self.data_val = random_split(
            data_full,
            [n_trn, n_val],
            generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, 
            num_workers=self.num_trn_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, 
            num_workers=self.num_val_workers)


class LitVAE(pl.LightningModule):
    """
    Pytorch Lighting Wrapper of our VAE model for convenient
    handling
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = ChlVAE(cfg)
        self.cfg = cfg

    def forward(self, inputs: List[Tensor]) -> Tensor:
        """
        forward is responsible for inference-exclusive computations
        only.
        """
        return self.model(*inputs)

    def training_step(
        self, 
        batch: List[Tensor],
        batch_idx: int):

        results = self.forward(batch)
        train_loss = self.model.loss_function(
            results,
            batch,
            M_N=self.cfg.batch_size
        )

        for k, v in train_loss.items():
            # self.logger.experiment.add_scalar(k, v.item(), batch_idx)
            self.log(k, v.item())#  on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = self.model.loss_function(
            results, batch, M_N=self.cfg.batch_size)
        val_loss = {"val_lossR": loss['Reconstruction_Loss']}
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_lossR'] for x in outputs]).mean()
        # print(f"avg_loss: {avg_loss.item():.2f}")
        self.log('val_lossR', avg_loss.item())


    def configure_optimizers(self):
        optims = []
        scheds = []
        c = self.cfg

        optimizer = optim.Adam(self.model.parameters(),
                               lr=c.learning_rate,
                               weight_decay=c.weight_decay)
        optims.append(optimizer)
        return optims
        # More than 1 optimizer used for adversarial training
        """
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
        """

"""
////////////////
    def __init__(
        self, 
        dataset: Dataset,
        model: ChlVAE,
        cfg: Config):
        super(LitVAE, self).__init__()


        self.vae_model = model 

        self.dataset = dataset
        self.dloader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)
        self.model = model

    # def forward(self, input: List[Tensor]) -> List[Tensor]:

    def training_step(self, batch, batch_idx):
        dat, missing_mask = batch
        results = self.model(dat, missing_mask)
        self.model.lo
"""