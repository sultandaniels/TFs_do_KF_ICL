import collections
import copy
import gc
import logging
import math
import os
import pickle
import time
import re
import copy

import sys
notebook_dir = os.getcwd()
grandparent_dir = os.path.dirname(os.path.dirname(notebook_dir))
sys.path.append(grandparent_dir)
print(sys.path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from tensordict import TensorDict
from pandas.plotting import table
from datetime import datetime

from core import Config
from models import GPT2, CnnKF
from data_train import set_config_params
from create_plots_with_zero_pred import tf_preds
from linalg_helpers import print_matrix
from predictors import getMats, getSims

# def applyFuncToCkpts(func, model_name="ident", debug=False):
#     ckpt_dir = f"/data/shared/ICL_Kalman_Experiments/model_checkpoints/GPT2"
#     for ckpt_name in os.listdir(ckpt_dir):
#         ckpt_folder = os.path.join(ckpt_dir, ckpt_name)
#         if not os.path.isdir(ckpt_folder):
#             continue
#         else:
#             for ckpt_step in os.listdir(ckpt_folder):
#                 # make sure final file is valid checkpoint file
#                 ckpt_path = re.search(r"^step=(\d+)\.ckpt$", ckpt_step)
#                 if ckpt_path:
#                     ckpt_path = ckpt_path.group(1)
#                     generateCkptPlots(func, ckpt_path, model_name, debug)


# loops through all checkpoints in identity model
def getPredsEx(device, multi_sys_ys, nx=5, num_sys_haystack=2, model_name="ident", max_step = 17600, debug=False):
    model_preds = []
    ckpt_lst = []
    ckpt_base = f"/data/shared/ICL_Kalman_Experiments/model_checkpoints/GPT2/250124_052617.8dd0f8_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/checkpoints/step="
    for ckpt_step in range(100, max_step, 100):
        ckpt_path = ckpt_base + str(ckpt_step) + ".ckpt"

        if not os.path.exists(ckpt_path):
            if debug:
                print(f"Ckpt path {ckpt_path} not valid.")
            continue

        if debug:
            print(f"Ckpt path {ckpt_path} found.")
        
        # init config, params
        config = Config()
        output_dir, ckpt_dir, experiment_name = set_config_params(config, model_name)
        num_gpu = len(config.devices)
        batch_size = config.batch_size

        if debug:
            print(f"Checkpoint Path: {ckpt_path}")
            print(f"Output Directory: {output_dir}")
            print(f"Checkpoint Directory: {ckpt_dir}")
            print(f"Batch size: {batch_size}")
            print(f"Number of GPUs: {num_gpu}")
            print(f"Number of training examples: {ckpt_step*batch_size*num_gpu}")

        model = GPT2.load_from_checkpoint(config.ckpt_path,
                                        n_dims_in=config.n_dims_in, 
                                        n_positions=config.n_positions,
                                        n_dims_out=config.n_dims_out, 
                                        n_embd=config.n_embd,
                                        n_layer=config.n_layer, 
                                        n_head=config.n_head, 
                                        use_pos_emb=config.use_pos_emb, 
                                        map_location=device, strict=True).eval().to(device)

        if debug:
            print(f"model: {model}")

        #get interleaved test data
        config.override("n_positions", num_sys_haystack*12 + 12)

        preds_tf = tf_preds(multi_sys_ys, model, device, config)
        num_ex = ckpt_step*batch_size*num_gpu
        model_preds.append(copy.deepcopy(preds_tf))
        ckpt_lst.append(num_ex)

    return np.asarray(model_preds), np.asarray(ckpt_lst)