import math
from pytorch_lightning.callbacks import ModelCheckpoint

class LogScaleCheckpoint(ModelCheckpoint):
    def __init__(self, base=10, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_step = trainer.global_step
        if current_step > 0 and (math.log10(current_step) % 1 == 0):
            self.save_checkpoint(trainer, pl_module)