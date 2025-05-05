import logging

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from core import Config, training
from models import GPT2
from datasources import FilterDataset, DataModuleWrapper
import os
import time
import pickle
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.utilities.exceptions import _TunerExitException
import json
import torch

##DO NOT MIX pytorch_lightning with lightning.pytorch in the import statements (it causes weird bugs with the boiler plate code)

def pl_lr_finder(config, model, trainer, datamodule):
    #find the learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)


    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    # Open a file in write mode
    with open(parent_parent_dir + "/lr_finder_results.json", 'w') as file:
        # Write the print statement to the file
        json.dump(lr_finder.results, file, indent=4)
    
    fig = lr_finder.plot(suggest=True)
    fig.show()
    new_lr = lr_finder.suggestion()
    model.learning_rate = new_lr

    with open(parent_parent_dir + "/learning_rate.txt", 'w') as file:
        # Write the print statement to the file
        file.write(f"Learning rate suggestion: {str(new_lr)}")
        file.close()
    return new_lr

def train_gpt2(model, config, ckpt_dir, train_mix_dist=False, train_mix_state_dim=False): #input emd_dim as a parameter for the embed dim experiment plots
    # a function to train GPT2 model

    torch.set_float32_matmul_precision('high') #set the matmul performance for Tensor Cores (or 'medium' depending on your precision-performance trade-off preference)

    logger = logging.getLogger(__name__)
    print("batch_size:", config.batch_size)
    print("train_steps:", config.train_steps)
    print("number of epochs:", config.num_epochs)
    print("context length:", config.n_positions)
    print("num_tasks:", config.num_tasks)

    

    #for BLISS server
    main_dir = f"/data/shared/ICL_Kalman_Experiments/train_and_test_data"

    val_dset = FilterDataset(main_dir + f"/{config.val_dataset_typ}/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", use_true_len=True) if os.path.exists(main_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl") else None

    datamodule = DataModuleWrapper(config, FilterDataset(main_dir + f"/{config.dataset_typ}/train_{config.dataset_typ}{config.C_dist}" + f"_state_dim_{config.nx}" + ("_dist_mix" if train_mix_dist else "") + ("_state_dim_mix" if train_mix_state_dim else "") + ".pkl"), val_dset)

    # Define model
    # output_dir = training.setup_train(model)

    print("training data dir:", main_dir + f"/{config.dataset_typ}/train_{config.dataset_typ}{config.C_dist}" + f"_state_dim_{config.nx}" + ("_dist_mix" if train_mix_dist else "") + ("_state_dim_mix" if train_mix_state_dim else "") + ".pkl")

    
    callbacks, loggers = training.get_callbacks_and_loggers(config, ckpt_dir, config.train_int)
    ckpt_path = config.ckpt_path if config.ckpt_path != '' else None
    print("ckpt_path:", config.ckpt_path)
    
    # trainer = pl.Trainer(
    #     accelerator="gpu",
    #     callbacks=callbacks,
    #     logger=loggers,
    #     gradient_clip_algorithm=config.gradient_clip_algorithm,
    #     gradient_clip_val=config.gradient_clip_val,
    #     log_every_n_steps=50,
    #     max_epochs=config.num_epochs
    # )

    wandb_logger = WandbLogger(log_model="all")
    trainer = pl.Trainer(
        fast_dev_run=False,
        accelerator="gpu",
        devices=config.devices,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.train_int,
        min_epochs=config.num_epochs if config.use_true_len else 0,
        max_steps=-1 if config.use_true_len else config.train_steps,
        accumulate_grad_batches=config.acc_grad_batch,
        # max_epochs=config.num_epochs,
        strategy=DDPStrategy(find_unused_parameters=True) #only for BLISS GPUs
    )

    if config.learning_rate == 0.0:
        # find learning rate with pytorch lightning
        new_lr = pl_lr_finder(config, model, trainer, datamodule)
        print("suggested learning rate:", new_lr)

    # time how long it takes to train the model
    time_start = time.time()
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file {ckpt_path} does not exist.")
        # os.makedirs(ckpt_path, exist_ok=True)
        # Handle the situation, e.g., by aborting the program, loading a different checkpoint, etc. 
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    time_end = time.time()
    return time_end - time_start

if __name__ == '__main__':
    # #load the numpy folder in ../data and unpack the data.pkl file and show the keys
    # # Load the pickle file
    # with open('../data/numpy/data.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # # If the data is a dictionary, print its keys
    # if isinstance(data, dict):
    #     print("data[observation].shape:", data["observation"].shape)
    #     print("data[state].shape:", data["state"].shape)
    # else:
    #     print("The loaded data is not a dictionary.")

    # with open("../data/val_ypred.pkl", "rb") as f:
    #         entries = pickle.load(f)
    #         print("keys of entries:", entries[0].keys())
    #         print("len of entries:", len(entries))
    #         print("shape of all the values for each key in entries[0]", {k: v.shape for k, v in entries[0].items()})
    #         # print("shape of entries:", entries["observation"].shape)

    config = Config()
    model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                 n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
    train_gpt2(model, config)
    




