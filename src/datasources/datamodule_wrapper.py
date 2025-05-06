import pytorch_lightning as pl
from core import Config
from torch.utils.data import DataLoader
import torch

config = Config()


class DataModuleWrapper(pl.LightningDataModule):
    def __init__(self, config, train_ds, val_ds=None):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

    def custom_collate_fn(self, batch):
        # Assume all dicts have the same keys
        elem = batch[0]
        collated = {}
        for key in elem:
            if isinstance(elem[key], torch.Tensor):
                collated[key] = torch.stack([d[key] for d in batch])
            else:
                # Keep as list (do not stack)
                collated[key] = [d[key] for d in batch]
        return collated

    def train_dataloader(self):

        if config.mem_suppress:
            if config.masking or not (config.cached_data or config.masking):
                return DataLoader(
                    self.train_ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.train_data_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=self.custom_collate_fn
                )
            elif not config.masking and config.cached_data:
                return DataLoader(
                    self.train_ds,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.train_data_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    drop_last=False
                )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.train_data_workers,
                persistent_workers=True,
                pin_memory=True,
                drop_last=False
            )


    # def val_dataloader(self):
    #     if self.val_ds is not None:
    #         return DataLoader(
    #             self.val_ds,
    #             batch_size=config.test_batch_size,
    #             shuffle=False,
    #             num_workers=config.test_data_workers,
    #             persistent_workers=True,
    #             pin_memory=True,
    #             drop_last=False,
    #         )
