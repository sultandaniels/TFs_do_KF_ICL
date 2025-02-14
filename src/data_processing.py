import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
print(os.getcwd())
import time
import gc
import torch
import bisect

#import empirical cdf
import sys
# sys.path.append(os.path.abspath('streamlined_mop/src'))
# sys.path.append(os.path.abspath('..'))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_val_data(valA, C_dist, exper, nx):
    path = f"../outputs/GPT2/{exper}/data/val_{valA}{C_dist}_state_dim_{nx}.pkl"
    obs = []
    if os.path.exists(path):
        print(os.path.abspath(path))
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for item in data:
            obs.append(item["obs"])
        del data

        print(f"loaded state dim {nx} validation obs")
        torch.cuda.empty_cache()
        gc.collect()
    else:
        raise ValueError(f"State dim {nx} validation data does not exist for this experiment")
    return obs

def gen_ckpt_steps(minval, maxval, interval):
    ckpt_steps = list(range(minval, maxval + 1, interval))
    return ckpt_steps

def move_dict_to_device(d, device):
    for key, value in d.items():
            d[key] = torch.Tensor(value).to(device)
    return d

def get_other_err(valA, C_dist, ckpt_step, exper, curve, nx, single_system=False, nope=False, zero_cut=False, train_conv=False):
    path = f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{exper}/prediction_errors{C_dist}_step={str(ckpt_step)}.ckpt/"+ ("train_conv_" if train_conv else "") + ("zero_cut_" if zero_cut else "") + ("single_system_" if single_system else "") + f"{valA}_state_dim_{nx}_err_lss.pkl"
    curve_err = None

    if os.path.exists(path):
        print(os.path.abspath(path))
        with open(path, 'rb') as f:
            err_lss = pickle.load(f)
            err_lss = move_dict_to_device(err_lss, device)
        if curve in err_lss:
            curve_err = err_lss[curve]
            print(f"loaded {curve} error")
        else:
            raise ValueError(f"{curve} Preds not in this checkpoint")
        del err_lss
        torch.cuda.empty_cache()
        gc.collect()
    else:
        raise ValueError(f"Preds do not exist for this checkpoint at:\n{path}")
    return curve_err

def get_mop_ratios_ckpt(valA, C_dist, ckpt_step, exper, nx, single_system=False, nope=False, zero_cut=False, train_conv=False):
    mop_err = None
    pred_ckpt = None
    #print the absolute path of the experiment

    path = f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{exper}/prediction_errors{C_dist}_step={str(ckpt_step)}.ckpt/"+ ("train_conv_" if train_conv else "") + ("zero_cut_" if zero_cut else "") + ("single_system_" if single_system else "") + f"{valA}_state_dim_{nx}_err_lss.pkl"
    if os.path.exists(path):
        print(os.path.abspath(path))
        #load prediction errors

        with open(path, 'rb') as f:
            err_lss = pickle.load(f)
            err_lss = move_dict_to_device(err_lss, device)

        mop_err = err_lss["MOP"]
        pred_ckpt = ckpt_step
        
        del err_lss
        torch.cuda.empty_cache()
        gc.collect()

        if not (mop_err == None):
            print("Loaded Transformer Errors")
        else:
            raise ValueError("TF Preds do not exist for this checkpoint")
    else:
        print(f"path does not exist: {path}")
    return mop_err, pred_ckpt

def compute_ratio(ind, err, kalman_err, single_system=False):
    
    if kalman_err is None:
        print(f"err shape: {err.shape}")
        ratios = err
    elif err.shape != kalman_err.shape:
        #take the reciprocal of every element in kalman_err
        rec_kalman = 1/kalman_err
        #multiply rec_kalman by analytical error
        irr_err = err[:,0]
        ratios = rec_kalman * irr_err[:,np.newaxis, np.newaxis]
    else:
        ratios = err/kalman_err[:,:,0:err.shape[-1]]

    # Compute the 25th, 50th, and 75th percentiles along axis 1
    device = ratios.device
    percentiles = torch.tensor([0.25, 0.5, 0.75], device=device)

    #take the median of the ratios along axis 1
    print(f"ratios shape: {ratios.shape}")

    if not single_system:
        ratios, _ = torch.median(ratios, axis=1)
    else:
        ratios = ratios[0]

    print(f"ratios shape after median: {ratios.shape}")

    if ind == None:
        ratios_percentiles = torch.quantile(ratios, percentiles, dim=0)

    else:
        ratios_percentiles = torch.quantile(ratios[:,ind], percentiles)
        # ratios_med = torch.median(ratios[:, ind])

    del ratios
    torch.cuda.empty_cache()
    gc.collect()
    
    return ratios_percentiles