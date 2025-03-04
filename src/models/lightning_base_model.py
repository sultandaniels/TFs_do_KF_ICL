import torch
import pytorch_lightning as pl
from core import Config

config = Config()


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate=config.learning_rate):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate

    def forward(self, input_dict, current_epoch=None):
        raise NotImplementedError("Base Class")

    def log_output_dct(self, output_dict, typ):
        for k in output_dict:
            if "loss" in k or "metric" in k:
                self.log(typ+"_"+k, output_dict[k], on_step=True, on_epoch=True,
                         prog_bar=True, logger=True, sync_dist=True) #added sync_dist=True for distributed training

        if hasattr(config, "loss_weighting_strategy") and config.loss_weighting_strategy in ["gradnorm", "my_gradnorm"] and typ == "train":
            for name, param in self.loss_weighting_strategy.lw_dict.items():
                self.log(name, param.detach().item(), on_step=True,
                         on_epoch=True, prog_bar=True, logger=True)

    def training_step(self, input_dict, batch_idx): 
        #Calls the model's forward method with the input data and batch index. The return_intermediate_dict=True argument indicates that the forward method should return both intermediate and final outputs. The intermediate dict holds the training predictions
        
        intermediate_dict, output_dict = self(
            input_dict, batch_idx=batch_idx, return_intermediate_dict=True)
        self.log_output_dct(output_dict, "train")
        return {"loss": output_dict["optimized_loss"],
                "intermediate_dict": intermediate_dict,
                "output_dict": output_dict}

    def on_after_backward(self):
        if hasattr(config, "loss_weighting_strategy") and config.loss_weighting_strategy in ["gradnorm", "my_gradnorm"]:
            self.loss_weighting_strategy.normalize_coeffs()

    def validation_step(self, input_dict, batch_idx):
        intermediate_dict, output_dict = self(
            input_dict, batch_idx=batch_idx, return_intermediate_dict=True)
        self.log_output_dct(output_dict, "val")
        return {"loss": output_dict["optimized_loss"],
                "intermediate_di]-[ct": intermediate_dict,
                "output_dict": output_dict}

    def test_step(self, input_dict, batch_idx):
        output_dict = self(input_dict, batch_idx=batch_idx)
        self.log_output_dct(output_dict, "test")

    def configure_optimizers(self):
        print("current learning rate:", self.learning_rate)
        print("config learning rate:", config.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, #changed from config.learning_rate
                                    weight_decay=config.weight_decay)
        return optimizer