import collections
import copy
import gc
import logging
import math
import os
import pickle
import time
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from tensordict import TensorDict
from pandas.plotting import table
from datetime import datetime

from core import Config
from dyn_models import apply_kf
from models import GPT2, CnnKF
from utils import RLS, plot_errs, plot_errs_conv, plot_errs_multi_sys
from datasources import filter_dataset
from datasources.filter_dataset import populate_traces, special_tokens
from collect_data import collect_data
import linalg_helpers as la

plt.rcParams['axes.titlesize'] = 20
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


####################################################################################################
# from wentinn's code
def wentinn_compute_errors(config):
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger
    config = Config()  # get the config

    num_systems = config.num_tasks["test"]
    num_trials = config.num_traces["test"]

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head).eval().to(
        device)  # load_from_checkpoint

    with open(f"../data/test_sim.pt", "rb") as f:
        sim_objs = torch.load(f)
    with open(f"../data/test_{config.val_dataset_typ}.pt", "rb") as f:
        samples = torch.load(f)
        ys = samples["obs"].numpy()

    # ys, us = [], []  # initialize the lists
    # sim_objs = []
    # for i in range(num_systems):  # iterate over 1000 (I think this is the number of trials for the dataset)
    #     if config.dataset_typ == "drone":  # if the dataset type is drone
    #         sim_obj, entry = generate_drone_sample(config.n_positions)  # generate drone sample
    #         us.append(entry["actions"])  # append the actions
    #     else:
    #         if config.changing:  # if the dataset is changing
    #             sim_obj, entry = generate_changing_lti_sample(config.n_positions, config.nx, config.ny,
    #                                                           n_noise=config.n_noise)  # generate changing lti sample
    #         else:
    #             sim_obj, entry = generate_lti_sample(config.dataset_typ,
    #                                                  num_trials,
    #                                                  config.n_positions,
    #                                                  config.nx, config.ny,
    #                                                  n_noise=config.n_noise)  # generate lti sample
    #     ys.append(entry["obs"])  # append the observations
    #     sim_objs.append(sim_obj)  # append the sim object
    # ys = torch.stack(ys).numpy()
    # us = torch.stack(us).numpy()

    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)
        # if config.dataset_typ == "drone":  # if the dataset type is drone
        #     I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

        if config.changing:
            preds_tf = model.predict_ar(ys[:, :-1])  # predict using the model
        else:
            batch_shape = I.shape[:-2]
            flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))

            _, flattened_preds_tf = model.predict_step(
                {"current": torch.from_numpy(flattened_I).to(device)})  # predict using the model
            preds_tf = np.reshape(flattened_preds_tf["preds"].cpu().numpy(),
                                  (*batch_shape, *I.shape[-2:]))  # get the predictions
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                      axis=-2)  # concatenate the predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    errs_zero = np.linalg.norm((ys - np.zeros_like(ys)), axis=-1) ** 2  # get the errors of zero predictions

    n_noise = config.n_noise

    # if config.dataset_typ == "drone":
    #     preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
    # else:
    #     preds_kf = np.array([[
    #             apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
    #             for __ys in _ys
    #         ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
    #     ])  # get kalman filter predictions
    preds_kf = np.array([[
        apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w * np.sqrt(n_noise), sigma_v=sim_obj.sigma_v * np.sqrt(n_noise))
        for __ys in _ys
    ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
    ])  # get kalman filter predictions
    errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2

    err_lss = collections.OrderedDict([
        ("Kalman", errs_kf),
        ("MOP", errs_tf),
        ("Zero", errs_zero)
    ])

    if config.dataset_typ != "drone":
        #     preds_rls = []
        #     preds_rls_analytical = []
        #     for sim_obj, _ys in zip(sim_objs, ys):
        #         _preds_rls = []
        #         _preds_rls_analytical = []
        #         for __ys in _ys:3
        #             ls = [np.zeros(config.ny)]
        #             ls_analytical = [np.linalg.norm(__ys[0], axis=-1) ** 2]

        #             rls = RLS(config.nx, config.ny)
        #             for i in range(_ys.shape[-2] - 1):
        #                 if i < ir_length:
        #                     ls.append(__ys[i])
        #                     ls_analytical.append(np.linalg.norm(__ys[i + 1], axis=-1) ** 2)
        #                 else:
        #                     rls.add_data(__ys[i - ir_length:i].flatten(), __ys[i])
        #                     _cnn_rls = CnnKF(config.ny, ir_length)
        #                     _cnn_rls.observation_IR.data = torch.from_numpy(np.stack([_rls.mu for _rls in rls.rlss], axis=-1)
        #                                                                     .reshape(ir_length, config.ny, config.ny)
        #                                                                     .transpose(1, 0, 2)[:, ::-1].copy())

        #                     ls.append(rls.predict(__ys[i - ir_length + 1:i + 1].flatten()))
        #                     ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())

        #             _preds_rls.append(ls)
        #             _preds_rls_analytical.append(ls_analytical)

        #         preds_rls.append(_preds_rls)
        #         preds_rls_analytical.append(_preds_rls_analytical)

        #     err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
        #     err_lss["OLS_analytical"] = np.array(preds_rls_analytical)

        # Debugging implemented OLS
        for ir_length in range(1, 4):
            print(f"IR length: {ir_length}")
            preds_rls_wentinn = []
            preds_rls_wentinn_analytical = []
            for sim_obj, _ys in zip(sim_objs, ys):
                _preds_rls_wentinn = []
                _preds_rls_wentinn_analytical = []
                for __ys in _ys:
                    padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])  # [(L + R - 1) x O_D]
                    ls = list(np.zeros((2, config.ny)))
                    ls_analytical = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)

                    rls_wentinn = CnnKF(config.ny, ir_length, ridge=1.0)
                    for i in range(config.n_positions - 1):
                        rls_wentinn.update(
                            torch.from_numpy(padded_ys[i:i + ir_length]),
                            torch.from_numpy(padded_ys[i + ir_length])
                        )

                        ls.append(rls_wentinn(torch.Tensor(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0,
                                                                                                              1).detach().numpy())
                        ls_analytical.append(rls_wentinn.analytical_error(sim_obj).item())

                    _preds_rls_wentinn.append(ls)
                    _preds_rls_wentinn_analytical.append(ls_analytical)

                preds_rls_wentinn.append(_preds_rls_wentinn)
                preds_rls_wentinn_analytical.append(_preds_rls_wentinn_analytical)

            err_lss[f"OLS_ir_length{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
            # err_lss[f"OLS_ir_length{ir_length}_analytical"] = np.array(preds_rls_wentinn_analytical)

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)

    plot_errs(err_lss, irreducible_error, ax=ax, shade=True, normalized=False)
    # plot_errs(err_lss, irreducible_error, ax=ax, shade=config.dataset_typ != "drone", normalized=True)

    # plot_errs(err_lss, irreducible_error, ax=ax, shade=True, normalized=False)
    # ax.plot(np.arange(config.n_positions + 1), np.full(config.n_positions + 1, np.mean(irreducible_error)), color='black', linewidth=5, linestyle='--')

    os.makedirs("../figures", exist_ok=True)
    fig.savefig(f"../figures/{config.val_dataset_typ}" + ("-changing" if config.changing else ""))
    plt.show()


####################################################################################################
def get_step_number(ckpt_path):
    # Extract the filename from the path
    filename = ckpt_path.split('/')[-1]
    
    # Use regex to find the step number
    match = re.search(r'step=(\d+)', filename)
    
    if match:
        return match.group(1)
    else:
        return None
    
def compute_OLS_and_OLS_analytical(config, ys, sim_objs, ir_length, err_lss):  # PENDING DELETION
    preds_rls = []
    preds_rls_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls = []
        _preds_rls_analytical = []
        for __ys in _ys:
            ls = [np.zeros(config.ny)]
            ls_analytical = [np.linalg.norm(__ys[0], axis=-1) ** 2]
            rls = RLS(config.nx, config.ny)

            # print("shape __ys:", __ys.shape)
            # print("range of for loop:", range(__ys.shape[-2] - 1))
            for i in range(_ys.shape[-2] - 1):
                if i < 3:
                    ls.append(__ys[i])
                    ls_analytical.append(np.linalg.norm(__ys[i + 1], axis=-1) ** 2)
                else:
                    if __ys[i - 3:i].shape[0] == 0:
                        print("i:", i)
                    rls.add_data(__ys[i - ir_length:i].flatten(), __ys[i])
                    _cnn_rls = CnnKF(config.ny, ir_length)
                    _cnn_rls.observation_IR.data = torch.from_numpy(np.stack([_rls.mu for _rls in rls.rlss], axis=-1)
                                                                    .reshape(ir_length, config.ny, config.ny)
                                                                    .transpose(1, 0, 2)[:, ::-1].copy())
                    ls.append(rls.predict(__ys[i - (ir_length - 1):i + 1].flatten()))
                    ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())
            _preds_rls.append(ls)
            _preds_rls_analytical.append(ls_analytical)

        preds_rls.append(_preds_rls)
        preds_rls_analytical.append(_preds_rls_analytical)

    err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
    err_lss["OLS_analytical"] = np.array(preds_rls_analytical)
    return err_lss


def compute_OLS_and_OLS_analytical_revised(config, ys, sim_objs, ir_length, err_lss):
    # Ensure PyTorch version supports MPS and MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")  # Fallback to CPU if MPS is not available
    preds_rls = []
    preds_rls_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls = []
        _preds_rls_analytical = []
        for __ys in _ys:
            # Convert numpy arrays to tensors and move to MPS
            __ys_tensor = torch.from_numpy(__ys).to(device)
            ls = [torch.zeros(config.ny, device=device)]
            ls_analytical = [torch.linalg.norm(__ys_tensor[0], axis=-1) ** 2]
            rls = RLS(config.nx, config.ny)  # Assuming RLS can handle MPS tensors
            for i in range(__ys_tensor.shape[-2] - 1):
                if i < 2:
                    ls.append(__ys_tensor[i])
                    ls_analytical.append(torch.linalg.norm(__ys_tensor[i + 1], axis=-1) ** 2)
                else:
                    rls.add_data_tensor(__ys_tensor[i - 2:i].flatten(), __ys_tensor[i])
                    _cnn_rls = CnnKF(config.ny, ir_length)
                    # Ensure _cnn_rls can handle MPS tensors
                    _cnn_rls.observation_IR.data = torch.stack([_rls.mu for _rls in rls.rlss], axis=-1).reshape(
                        ir_length, config.ny, config.ny).transpose(1, 0, 2)[:, ::-1].copy().to(device)
                    ls.append(rls.predict(__ys_tensor[i - 1:i + 1].flatten()))
                    ls_analytical.append(_cnn_rls.analytical_error(sim_obj).item())

            _preds_rls.append(ls)
            _preds_rls_analytical.append(ls_analytical)

        preds_rls.append(_preds_rls)
        preds_rls_analytical.append(_preds_rls_analytical)

    # Convert predictions back to CPU and numpy for final operations
    preds_rls = [torch.stack(pred).cpu().numpy() for pred in preds_rls]
    preds_rls_analytical = [torch.tensor(pred).cpu().numpy() for pred in preds_rls_analytical]

    err_lss["OLS"] = np.linalg.norm(ys - np.array(preds_rls), axis=-1) ** 2
    err_lss["OLS_analytical"] = np.array(preds_rls_analytical)
    return err_lss

def compute_OLS_ir(config, ys, sim_objs, max_ir_length, err_lss):

    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    # set torch precision to float64
    torch.set_default_dtype(torch.float64)
    # print("max_ir_length + 1:", max_ir_length + 1)
    for ir_length in range(1, max_ir_length + 1):
        start = time.time()
        print(f"\tIR length: {ir_length}")

        if ir_length == 2:
            preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper(config, ys, sim_objs, ir_length, 0.0)

            err_lss[f"OLS_ir_{ir_length}_unreg"] = np.linalg.norm(ys - np.array(preds_rls_wentinn.cpu()), axis=-1) ** 2
            err_lss[f"OLS_analytical_ir_{ir_length}_unreg"] = np.array(preds_rls_wentinn_analytical.cpu())

            del preds_rls_wentinn
            del preds_rls_wentinn_analytical
            torch.cuda.empty_cache()
            gc.collect()

        preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper(config, ys, sim_objs, ir_length, 1.0)

        err_lss[f"OLS_ir_{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn.cpu()), axis=-1) ** 2
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.array(preds_rls_wentinn_analytical.cpu())
        end = time.time()
        print("\ttime elapsed:", (end - start) / 60, "min\n")

        del preds_rls_wentinn
        del preds_rls_wentinn_analytical   
        torch.cuda.empty_cache()
        gc.collect()

        # # Check if CUDA is available
        # if torch.cuda.is_available():

        #     # Print memory usage
        #     print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
        #     print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
        #     print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")
        #     print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 2):.2f} MB")
        # else:
        #     print("CUDA is not available.")
    # set torch precision back to float32
    torch.set_default_dtype(torch.float32)
    return err_lss

def compute_OLS_ir_multi_sys(num_trace_configs, next_start_per_config, seg_lens_per_config, sys_choices_per_config, sys_inds_per_config, sim_obs_per_config, config, ys, max_ir_length, err_lss):
    #ys are the observations from the original validation dataset before interleaving

    # set torch precision to float64
    torch.set_default_dtype(torch.float64)

    for ir_length in range(1, max_ir_length + 1):
        # Initialize the err_lss ols and ols analytical values to infinity shaped like the multi_sys_ys
        err_lss[f"OLS_ir_{ir_length}"] = np.full((num_trace_configs, ys.shape[1], ys.shape[2]), np.inf)
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.full((num_trace_configs, ys.shape[1], ys.shape[2]), np.inf)

        # if ir_length == 2:
        #     err_lss[f"OLS_ir_{ir_length}_unreg"] = np.full(multi_sys_ys.shape[:-1], np.inf)
        #     err_lss[f"OLS_analytical_ir_{ir_length}_unreg"] = np.full(multi_sys_ys.shape[:-1], np.inf)

    

    for trace_conf in range(num_trace_configs):
        # print(f"\n\nTrace config: {trace_conf}")
        # print(f"sys_inds_per_config[trace_conf]: {sys_inds_per_config[trace_conf]}")
        # print(f"sim_obs_per_config[trace_conf]: {sim_obs_per_config[trace_conf]}")
        # print(f"sim_obs_per_config[trace_conf].values(): {sim_obs_per_config[trace_conf].values()}")
        # print(f"type of sim_obs_per_config[trace_conf].values(): {type(list(sim_obs_per_config[trace_conf].values()))}")

        sim_objs_list = list(sim_obs_per_config[trace_conf].values())
        # print(seg_lens_per_config[trace_conf])
        #create a new array named ys_sys that takes the ys from axis 0 that correspond to every index in sys_inds_per_config[trace_conf]
        ys_sys = ys[sys_inds_per_config[trace_conf]]
        # print(f"ys_sys shape: {ys_sys.shape}, len of sys_inds_per_config[trace_conf]: {len(sys_inds_per_config[trace_conf])}, ys shape: {ys.shape}")

        ols_err_lss = {}
        ols_err_lss = compute_OLS_ir(config, ys_sys, sim_objs_list, max_ir_length, ols_err_lss)

        sys_start = {sys_ind: 0 for sys_ind in sys_inds_per_config[trace_conf]} #initialize the start of the segment for each system
        seg_count = 0
        for next_start in next_start_per_config[trace_conf]:
            sys = sys_choices_per_config[trace_conf][seg_count]

            #find the index of sys in sys_inds_per_config[trace_conf]
            sys_ind = sys_inds_per_config[trace_conf].index(sys)

            seg_len = seg_lens_per_config[trace_conf][seg_count] # get the length of the segment

            for ir_length in range(1, max_ir_length + 1):
                # print(f"ir_length: {ir_length}")
                # print(f"sys_start[sys]: {sys_start[sys]}")
                # print(f"sys_start[sys] + seg_len: {sys_start[sys] + seg_len}")
                # print(f"next_start + 1: {next_start + 1}")
                # print(f"next_start + 1 + seg_len: {next_start + 1 + seg_len}")
                # print(f"ols_err_lss[f'OLS_ir_{ir_length}'][sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len].shape: {ols_err_lss[f'OLS_ir_{ir_length}'][sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len].shape}")
                # print(f"err_lss[f'OLS_ir_{ir_length}'][trace_conf, :, next_start + 1:next_start + 1 + seg_len].shape: {err_lss[f'OLS_ir_{ir_length}'][trace_conf, :, next_start + 1:next_start + 1 + seg_len].shape}")
                err_lss[f"OLS_ir_{ir_length}"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = ols_err_lss[f"OLS_ir_{ir_length}"][sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len]
                err_lss[f"OLS_analytical_ir_{ir_length}"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = ols_err_lss[f"OLS_analytical_ir_{ir_length}"][sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len]
                
            sys_start[sys] += seg_len
            seg_count += 1

    return err_lss

def compute_OLS_needle(num_trace_configs, next_start_per_config, seg_lens_per_config, sys_choices_per_config, sys_inds_per_config, sim_obs_per_config, config, ys, errs_all, max_ir_length, err_lss):
    #ys are the observations from the original validation dataset before interleaving

    # set torch precision to float64
    torch.set_default_dtype(torch.float64)

    for ir_length in range(1, max_ir_length + 1):
        # Initialize the err_lss ols and ols analytical values to infinity shaped like the multi_sys_ys
        err_lss[f"OLS_ir_{ir_length}"] = np.full((num_trace_configs, ys.shape[1], ys.shape[2]), np.inf)
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.full((num_trace_configs, ys.shape[1], ys.shape[2]), np.inf)

    print("interleaving OLS_errors")
    for trace_conf in range(num_trace_configs):

        sys_start = {sys_ind: 0 for sys_ind in sys_inds_per_config[trace_conf]} #initialize the start of the segment for each system
        seg_count = 0
        for next_start in next_start_per_config[trace_conf]:
            sys = sys_choices_per_config[trace_conf][seg_count]

            #find the index of sys in sys_inds_per_config[trace_conf]
            sys_ind = sys_inds_per_config[trace_conf].index(sys)

            seg_len = seg_lens_per_config[trace_conf][seg_count] # get the length of the segment

            for ir_length in range(1, max_ir_length + 1):
                err_lss[f"OLS_ir_{ir_length}"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_all[f"OLS_ir_{ir_length}"][sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len]
                err_lss[f"OLS_analytical_ir_{ir_length}"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_all[f"OLS_analytical_ir_{ir_length}"][sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len]
                
            sys_start[sys] += seg_len
            seg_count += 1

    return err_lss



def compute_OLS_helper(config, ys, sim_objs, ir_length, ridge):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    n_positions = ys.shape[-2] - 1
    torch.set_default_device(device)
    preds_rls_wentinn = torch.zeros(np.expand_dims(ys[0],axis=0).shape)
    preds_rls_wentinn_analytical = torch.zeros(np.expand_dims(ys[0,:,:,0], axis=0).shape)


    with torch.no_grad():
        for sys in range(ys.shape[0]): # iterate over the number of systems
            ys_sys = np.expand_dims(ys[sys], axis=0)
        
            # [n_systems x n_traces x (n_positions + 1) x O_D]
            torch_ys = torch.Tensor(ys_sys).to(device)


            # [n_systems x n_traces x (n_positions + ir_length) x O_D]
            padded_ys = torch.cat([
                torch.zeros((*torch_ys.shape[:2], ir_length - 1, config.ny)).to(device), torch_ys.to(device)
            ], dim=-2)

            Y_indices = torch.arange(ir_length, (n_positions - 1) + ir_length)[:, None]  # [(n_positions - 1) x 1]
            X_indices = Y_indices - 1 - torch.arange(ir_length)

            del torch_ys
            torch.cuda.empty_cache()
            gc.collect()

            X, Y = padded_ys[..., X_indices, :], padded_ys[..., Y_indices, :]   # [n_systems x n_traces x (n_positions - 1) x ir_length x O_D], [n_systems x n_traces x (n_positions - 1) x 1 x O_D]

            flattened_X, flattened_Y = X.flatten(-2, -1), Y.flatten(-2, -1)     # [n_systems x n_traces x (n_positions - 1) x (ir_length * O_D)], [n_systems x n_traces x (n_positions - 1) x O_D]

            # [n_systems x n_traces x (n_positions - 1) x (ir_length * I_D) x (ir_length * O_D)]
            cumulative_XTX = torch.cumsum(flattened_X[..., :, None].to(device) * flattened_X[..., None, :], dim=-3).to(device) + ridge * torch.eye(ir_length * config.ny).to(device)
            # [n_systems x n_traces x (n_positions - 1) x (ir_length * I_D) x O_D]
            cumulative_XTY = torch.cumsum(flattened_X[..., :, None] * flattened_Y[..., None, :], dim=-3)

            min_eqs = config.ny if ridge == 0.0 else 1

            _rank_full = torch.inverse(cumulative_XTX[..., min_eqs - 1:, :, :]) @ cumulative_XTY[..., min_eqs - 1:, :, :]
            _rank_deficient = []
            for n_eqs in range(1, min_eqs):
                _rank_deficient.append(torch.linalg.pinv(flattened_X[..., :n_eqs, :]) @ flattened_Y[..., :n_eqs, :])
            if len(_rank_deficient) == 0:
                _rank_deficient = torch.zeros_like(_rank_full[..., :0, :, :])
            else:
                _rank_deficient = torch.stack(_rank_deficient, dim=-3)

            # [n_systems x n_traces x (n_positions - 1) x (ir_length * O_D) x O_D]
            # -> [n_systems x n_traces x (n_positions - 1) x ir_length x O_D x O_D]
            # -> [n_systems x n_traces x (n_positions - 1) x O_D x ir_length x O_D]
            observation_IRs = torch.cat([_rank_deficient, _rank_full], dim=-3).unflatten(-2, (ir_length, config.ny)).transpose(dim0=-3, dim1=-2)

            # SECTION: Compute the empirical output
            shifted_X = padded_ys[..., X_indices + 1, :]    # [n_systems x n_traces x (n_positions - 1) x ir_length x O_D]

            # Clean up padded_ys to free memory
            del padded_ys
            torch.cuda.empty_cache()
            gc.collect()

            flattened_observation_IRs = observation_IRs.flatten(0, 2)   # [B x O_D x ir_length x O_D]
            flattened_shifted_X = shifted_X.flatten(0, 2)               # [B x ir_length x O_D]

            # [n_systems x n_traces x (n_positions + 1) x O_D]
            torch_ys = torch.Tensor(ys_sys).to(device)

            del ys_sys
            torch.cuda.empty_cache()
            gc.collect()

            preds_rls_wentinn_sys = torch.vmap(Fn.conv2d)(
                flattened_observation_IRs,                                              # [B x O_D x ir_length x O_D]
                flattened_shifted_X.transpose(dim0=-2, dim1=-1)[..., None, :, :, None]  # [B x 1 x O_D x ir_length x 1]
            ).reshape(*torch_ys.shape[:2], n_positions - 1, config.ny) # [n_systems x n_traces x (n_positions - 1)]

            preds_rls_wentinn_sys = torch.cat([
                torch.zeros_like(torch_ys[..., :2, :]),
                preds_rls_wentinn_sys
            ], dim=-2)  # [n_systems x n_traces x (n_positions + 1) x O_D]

            if torch.all(preds_rls_wentinn == 0):
                preds_rls_wentinn = preds_rls_wentinn_sys
            else:
                preds_rls_wentinn = torch.vstack((preds_rls_wentinn, preds_rls_wentinn_sys))

            sim_objs_td = TensorDict({
                "F": torch.Tensor(np.stack([
                    sim_obj.A for sim_obj in sim_objs
                ], axis=0)),
                "H": torch.Tensor(np.stack([
                    sim_obj.C for sim_obj in sim_objs
                ], axis=0)),
                "sqrt_S_W": torch.stack([
                    torch.eye(config.nx) * sim_obj.sigma_w for sim_obj in sim_objs
                ]),
                "sqrt_S_V": torch.stack([
                    torch.eye(config.ny) * sim_obj.sigma_v for sim_obj in sim_objs
                ]),
            }, batch_size=(len(sim_objs),)).to(device)

            # SECTION: Compute analytical errors
            preds_rls_wentinn_analytical_sys = CnnKF.analytical_error(
                observation_IRs,            # [n_systems x n_traces x (n_positions - 1) x ...]
                sim_objs_td[sys, None, None]  # [n_systems x 1 x 1 x ...]
            )   # [n_systems x n_traces x (n_positions - 1)]

            preds_rls_wentinn_analytical_sys = torch.cat([
                torch.norm(torch_ys[..., :2, :], dim=-1) ** 2,    # [n_systems x n_traces x 2]
                preds_rls_wentinn_analytical_sys,               # [n_systems x n_traces x (n_positions - 1)]
            ], dim=-1)  # [n_systems x n_traces x (n_positions + 1)]

            # preds_rls_wentinn_analytical[sys] = preds_rls_wentinn_analytical_sys
            if torch.all(preds_rls_wentinn_analytical == 0):
                preds_rls_wentinn_analytical = preds_rls_wentinn_analytical_sys
            else:
                preds_rls_wentinn_analytical = torch.vstack((preds_rls_wentinn_analytical, preds_rls_wentinn_analytical_sys))

            del torch_ys
            torch.cuda.empty_cache()
            gc.collect()

    return preds_rls_wentinn, preds_rls_wentinn_analytical

def compute_OLS_ir_current(config, ys, sim_objs, max_ir_length, err_lss):
    # set torch precision to float64
    torch.set_default_dtype(torch.float64)
    print("\n\n max_ir_length + 1:", max_ir_length + 1)
    for ir_length in range(1, max_ir_length + 1):
        start = time.time()
        print(f"\n\nIR length: {ir_length}")

        if ir_length == 2:
            preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper_current(config, ys, sim_objs, ir_length, 0.0)

            err_lss[f"OLS_ir_{ir_length}_unreg"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
            err_lss[f"OLS_analytical_ir_{ir_length}_unreg"] = np.array(preds_rls_wentinn_analytical)

        preds_rls_wentinn, preds_rls_wentinn_analytical = compute_OLS_helper_current(config, ys, sim_objs, ir_length, 1.0)

        err_lss[f"OLS_ir_{ir_length}"] = np.linalg.norm(ys - np.array(preds_rls_wentinn), axis=-1) ** 2
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.array(preds_rls_wentinn_analytical)
        end = time.time()
        print("time elapsed:", (end - start) / 60, "min")
    # set torch precision back to float32
    torch.set_default_dtype(torch.float32)
    return err_lss


def compute_OLS_little_helper_current(ls, ls_analytical, sim_obj, padded_ys, ir_length, config, ridge):
    rls_wentinn = CnnKF(config.ny, ir_length, ridge=ridge)
    for i in range(config.n_positions - 1):
        obs_tensor = rls_wentinn.update(
            torch.from_numpy(padded_ys[i:i + ir_length]),
            torch.from_numpy(padded_ys[i + ir_length])
        )

        ls.append(
            rls_wentinn(torch.from_numpy(padded_ys[i + 1:i + ir_length + 1])[None]).squeeze(0, 1).detach().numpy())
        ls_analytical.append(rls_wentinn.analytical_error(sim_obj).item())

        # assert ls_analytical[-1] >= torch.trace(sim_obj.S_observation_inf).item(), f"Analytical error is less than irreducible error: {ls_analytical[-1]} < {torch.trace(sim_obj.S_observation_inf).item()}."
    return ls, ls_analytical


def compute_OLS_helper_current(config, ys, sim_objs, ir_length, ridge):
    preds_rls_wentinn = []
    preds_rls_wentinn_analytical = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _preds_rls_wentinn = []
        _preds_rls_wentinn_analytical = []
        for __ys in _ys:
            padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])  # [(L + R - 1) x O_D]
            ls = list(np.zeros((2, config.ny)))
            ls_analytical = list(np.linalg.norm(__ys[1], axis=-1) ** 2)

            ls, ls_analytical = compute_OLS_little_helper_current(ls, ls_analytical, sim_obj, padded_ys, ir_length, config,
                                                          ridge)

            _preds_rls_wentinn.append(ls)
            _preds_rls_wentinn_analytical.append(ls_analytical)

        preds_rls_wentinn.append(_preds_rls_wentinn)
        preds_rls_wentinn_analytical.append(_preds_rls_wentinn_analytical)
    return preds_rls_wentinn, preds_rls_wentinn_analytical


def compute_OLS_wentinn(config, ys, sim_objs, ir_length, err_lss):
    errs_rls_wentinn = []
    for sim_obj, _ys in zip(sim_objs, ys):
        _errs_rls_wentinn = []
        for __ys in _ys:
            padded_ys = np.vstack([np.zeros((ir_length - 1, config.ny)), __ys])  # [(L + R - 1) x O_D]
            ls = list(np.linalg.norm(__ys[:2], axis=-1) ** 2)
            rls_wentinn = CnnKF(config.ny, ir_length)
            for i in range(config.n_positions - 1):
                rls_wentinn.update(
                    torch.from_numpy(padded_ys[i:i + ir_length]),
                    torch.from_numpy(padded_ys[i + ir_length])
                )
                ls.append(rls_wentinn.analytical_error(sim_obj).item())
            _errs_rls_wentinn.append(ls)
        errs_rls_wentinn.append(_errs_rls_wentinn)
    err_lss["OLS_wentinn"] = np.array(errs_rls_wentinn)
    return err_lss


def batch_trace(x):
    # Ensure x has at least two dimensions
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least two dimensions")
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

def compute_kf(ys, sim_objs):
        print("start kf pred")
        preds_kf = np.array([[
            apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w,
                    sigma_v=sim_obj.sigma_v)
            for __ys in _ys
        ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
        ])  # get kalman filter predictions
        errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2
        return errs_kf

def compute_analytical_kf_simulation(config, ys, sim_objs, num_trials):

    n_positions = ys.shape[-2] - 1

    # Analytical Kalman Predictions
    analytical_kf = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    analytical_kf = analytical_kf.reshape((len(sim_objs), 1)) @ np.ones((1, n_positions))
    

    #Analytical simulation predictions
    #generate n_positions multivariate normal random variables with mean zero and covariance sim_obj.S_observation_inf and do this config.num_traces["val"] times for each sim_obj
    an_sims = np.array([np.random.multivariate_normal(np.zeros(config.ny), sim_obj.S_observation_inf, (num_trials, n_positions+1)) for sim_obj in sim_objs])
    an_sims = np.linalg.norm(an_sims, axis=-1) ** 2

    return analytical_kf, an_sims


def compute_errors(config, C_dist, run_deg_kf_test, wentinn_data, tf):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger

    print("val_dataset_typ:", config.val_dataset_typ)
    num_systems = config.num_val_tasks  # number of validation tasks
    print("Number of validation systems:", num_systems)
    num_trials = config.num_traces["val"]  # number of traces
    print("Number of traces:", num_trials)

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head, map_location=device, use_pos_embd=config.use_pos_embd).eval().to(
        device)  # load_from_checkpoint

    if wentinn_data:

        with open(f"../data/numpy_three_sys" + C_dist + "/test_sim.pt", "rb") as f:
            sim_objs = torch.load(f)

        with open('../data/numpy_three_sys' + C_dist + '/data.pkl',
                  'rb') as f:  # load the data.pkl file for the test data
            data = pickle.load(f)
            ys = data["observation"]
            print("ys.shape:", ys.shape)
    else:
        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)

        print("getting the validation data")
        # open fsim file
        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
            samples = pickle.load(f)
            # for every 2000 entries in samples, get the observation values and append them to the ys list
            ys = np.stack(
                [entry["obs"] for entry in samples], axis=0
            ).reshape((num_systems, num_trials, config.n_positions + 1, config.ny)).astype(np.float32)

            # prev_xs = np.concatenate([
            #     np.zeros((num_systems, num_trials, 1, config.nx)),
            #     np.stack(
            #         [entry["states"][:-1] for entry in samples], axis=0
            #     ).reshape((num_systems, num_trials, config.n_positions, config.nx)).astype(np.float32)
            # ], axis=2)
            # noiseless_ys = prev_xs @ np.stack([sim_obj.C @ sim_obj.A for sim_obj in sim_objs], axis=0)[:, None].transpose(0, 1, 3, 2)

            gc.collect()  # Start the garbage collector

    ckpt_steps = get_step_number(config.ckpt_path)
    print("ckpt_steps:", ckpt_steps)

    # if parent_parent_dir + f"/prediction_errors{config.C_dist}_step={str(ckpt_step)}.ckpt exists, load the prediction errors
    if os.path.exists(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl"):
        print("\nerr_lss already exists\n")
        with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'rb') as f:
            err_lss = pickle.load(f)

        for key in err_lss.keys():
            print(key, err_lss[key].shape)
            print("\n\n")
    else:
        print("\n err_lss does not exist yet")
        err_lss = collections.OrderedDict()


    # print("no tf pred")
    # Transformer Predictions
    if not ("MOP" in err_lss.keys()):
        print("\nstart tf pred")
        start = time.time()  # start the timer for transformer predictions
        print(f"ys.shape: {ys.shape}")
        with torch.no_grad():  # no gradients
            I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)
            # if config.dataset_typ == "drone":  # if the dataset type is drone
            #     I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

            if config.changing:
                preds_tf = model.predict_ar(ys[:, :-1])  # predict using the model
            else:
                # print("before model.predict_step()")
                batch_shape = I.shape[:-2]
                # print("batch_shape:", batch_shape)
                flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))
                # print("flattened_I.shape:", flattened_I.shape)
                validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I),
                                                                batch_size=config.test_batch_size)
                preds_arr = []  # Store the predictions for all batches
                for validation_batch in iter(validation_loader):
                    _, flattened_preds_tf = model.predict_step(
                        {"current": validation_batch.to(device)})  # .float().to(device)})    # predict using the model
                    preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
                preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                                    (*batch_shape, *I.shape[-2:]))  # Combine the predictions for all batches
                print("preds_tf.shape:", preds_tf.shape)
                preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                        axis=-2)  # concatenate the predictions
                print("preds_tf.shape:", preds_tf.shape)
        end = time.time()  # end the timer for transformer predictions
        print("time elapsed for MOP Pred:", (end - start) / 60, "min")  # print the time elapsed for transformer predictions

        errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
        err_lss["MOP"] = errs_tf

        os.makedirs(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt", exist_ok=True)
        with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'wb') as f:
                pickle.dump(err_lss, f)  
        
        del errs_tf
        del preds_tf

        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("TF pred already in err_lss")

    if tf: #only run transformer predictions
        irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
        return err_lss, irreducible_error
    
    # noiseless_errs_tf = np.linalg.norm((noiseless_ys - preds_tf), axis=-1) ** 2 + np.array([
    #     (np.linalg.norm(sim_obj.C) * sim_obj.sigma_w) ** 2 + config.ny * (sim_obj.sigma_v ** 2)
    #     for sim_obj in sim_objs
    # ])[:, None, None]
    # err_lss["Analytical_MOP"] = noiseless_errs_tf

    # del noiseless_errs_tf
    # del noiseless_ys

    print("start zero predictor")
    # zero predictor predictions
    errs_zero = np.linalg.norm(ys, axis=-1) ** 2  # get the errors of zero predictions
    err_lss["Zero"] = errs_zero

    os.makedirs(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt", exist_ok=True)
    with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'wb') as f:
            pickle.dump(err_lss, f)

    del errs_zero

    torch.cuda.empty_cache()
    gc.collect()



    if not ("Kalman" in err_lss.keys()):
        start = time.time()  # start the timer for kalman filter predictions
        if run_deg_kf_test:  # degenerate system KF Predictions

            #############################################################
            # this portion can most likely be deleted
            # Kalman Filter Predictions
            preds_kf_list = []
            for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)):
                inner_list = []
                for __ys in _ys:
                    result = apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w,
                                    sigma_v=sim_obj.sigma_v)
                    inner_list.append(result)
                preds_kf_list.append(inner_list)
            #############################################################

            preds_kf = np.array(preds_kf_list)  # get kalman filter predictions

            # create an array of zeros to hold the kalman filter predictions
            preds_kf = np.zeros((num_systems, num_systems, num_trials, config.n_positions + 1,
                                config.ny))  # first axis is the system that the kalman filter is being trained on, second axis is the system that the kalman filter is being tested on

            errs_kf = np.zeros((num_systems, num_systems, num_trials,
                                config.n_positions + 1))  # first axis is the system that the kalman filter is being trained on, second axis is the system that the kalman filter is being tested on
            # iterate over sim_objs
            kf_index = 0
            for sim_obj in sim_objs:  # iterate over the training systems
                for sys in range(num_systems):  # iterate over the test systems
                    print("Kalman filter", kf_index, "testing on system", sys)
                    for trial in range(num_trials):
                        preds_kf[kf_index, sys, trial, :, :] = apply_kf(sim_obj, ys[sys, trial, :-1, :],
                                                                        sigma_w=sim_obj.sigma_w,
                                                                        sigma_v=sim_obj.sigma_v)  # get the kalman filter predictions for the test system and the training system
                    errs_kf[kf_index, sys] = np.linalg.norm((ys[sys] - preds_kf[kf_index, sys]), axis=-1) ** 2  # get the errors of the kalman filter predictions for the test system and the training system
                kf_index += 1

        else:  
            # print("no kf pred")
            # Kalman Predictions
            print("start kf pred")
            preds_kf = np.array([[
                apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w,
                        sigma_v=sim_obj.sigma_v)
                for __ys in _ys
            ] for sim_obj, _ys in zip(sim_objs, np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2))
            ])  # get kalman filter predictions
            errs_kf = np.linalg.norm((ys - preds_kf), axis=-1) ** 2
            err_lss["Kalman"] = errs_kf

            os.makedirs(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt", exist_ok=True)
            with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'wb') as f:
                    pickle.dump(err_lss, f)

        end = time.time()  # end the timer for kalman filter predictions
        print("time elapsed for KF Pred:", (end - start) / 60,
            "min")  # print the time elapsed for kalman filter predictions


        del preds_kf
        del errs_kf

        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("Kalman pred already in err_lss")

    # # Check if CUDA is available
    # if torch.cuda.is_available():

    #     # Print memory usage
    #     print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    #     print(f"Memory Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
    #     print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")
    #     print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(device) / (1024 ** 2):.2f} MB")
    # else:
    #     print("CUDA is not available.")


    # Analytical Kalman Predictions
    analytical_kf = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    err_lss["Analytical_Kalman"] = analytical_kf.reshape((num_systems, 1)) @ np.ones((1, config.n_positions))
    

    #Analytical simulation predictions
    #generate config.n_positions multivariate normal random variables with mean zero and covariance sim_obj.S_observation_inf and do this config.num_traces["val"] times for each sim_obj
    an_sims = np.array([np.random.multivariate_normal(np.zeros(config.ny), sim_obj.S_observation_inf, (config.num_traces["val"], config.n_positions+1)) for sim_obj in sim_objs])

    # print("an_sims shape:", an_sims.shape)
    err_lss["Analytical_Simulation"] = np.linalg.norm(an_sims, axis=-1) ** 2

    os.makedirs(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt", exist_ok=True)
    with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'wb') as f:
            pickle.dump(err_lss, f)

    # print("no OLS")

    if not ("OLS" in err_lss.keys()):
        # Original OLS
        # Clear the PyTorch cache
        start = time.time()  # start the timer for OLS predictions
        print("start OLS pred")
        #print(torch.cuda.memory_summary())
        err_lss = compute_OLS_ir(config, ys, sim_objs, max_ir_length=3, err_lss=err_lss)


        os.makedirs(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt", exist_ok=True)
        with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'wb') as f:
                pickle.dump(err_lss, f)

        end = time.time()  # end the timer for OLS predictions
        print("time elapsed for OLS Pred:", (end - start) / 60, "min")  # print the time elapsed for OLS predictions
    else:
        print("OLS pred already in err_lss")

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])
    return err_lss, irreducible_error

def compute_errors_conv(config):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    print("val_dataset_typ:", config.val_dataset_typ)
    num_systems = config.num_val_tasks  # number of validation tasks
    print("Number of validation systems:", num_systems)
    num_trials = config.num_traces["val"]  # number of traces
    print("Number of traces:", num_trials)

    ckpt_steps = get_step_number(config.ckpt_path)
    print("ckpt_steps:", ckpt_steps)
    
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    
    # if parent_parent_dir + f"/prediction_errors{config.C_dist}_step={str(ckpt_step)}.ckpt exists, load the prediction errors
    if os.path.exists(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl"):
        print("\nerr_lss already exists\n")
        with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'rb') as f:
            err_lss = pickle.load(f)

        for key in err_lss.keys():
            print(key, err_lss[key].shape)
            print("\n\n")
            if key == "MOP":
                with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_irreducible_error.pkl", 'rb') as f:
                    irreducible_error = pickle.load(f)

                print("No need to redo MOP Predictions")
                return err_lss, irreducible_error
    else:
        print("\n err_lss does not exist yet")
        err_lss = collections.OrderedDict()

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head, map_location=device, use_pos_embd=config.use_pos_embd).eval().to(
        device)  # load_from_checkpoint

    print("getting the validation data")
    # open fsim file
    with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
        sim_objs = pickle.load(f)

    with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
        samples = pickle.load(f)
        # for every 2000 entries in samples, get the observation values and append them to the ys list
        ys = np.stack(
            [entry["obs"] for entry in samples], axis=0
        ).reshape((num_systems, num_trials, config.n_positions + 1, config.ny)).astype(np.float32)

        prev_xs = np.concatenate([
            np.zeros((num_systems, num_trials, 1, config.nx)),
            np.stack(
                [entry["states"][:-1] for entry in samples], axis=0
            ).reshape((num_systems, num_trials, config.n_positions, config.nx)).astype(np.float32)
        ], axis=2)
        noiseless_ys = prev_xs @ np.stack([sim_obj.C @ sim_obj.A for sim_obj in sim_objs], axis=0)[:, None].transpose(0, 1, 3, 2)

        gc.collect()  # Start the garbage collector

    # print("no tf pred")
    # Transformer Predictions
    print("\nstart tf pred")
    start = time.time()  # start the timer for transformer predictions
    with torch.no_grad():  # no gradients
        I = np.take(ys, np.arange(ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)
        # if config.dataset_typ == "drone":  # if the dataset type is drone
        #     I = np.concatenate([I, us], axis=-1)  # concatenate the inputs

        if config.changing:
            preds_tf = model.predict_ar(ys[:, :-1])  # predict using the model
        else:
            # print("before model.predict_step()")
            batch_shape = I.shape[:-2]
            # print("batch_shape:", batch_shape)
            flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))
            # print("flattened_I.shape:", flattened_I.shape)
            validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I),
                                                            batch_size=config.test_batch_size)
            preds_arr = []  # Store the predictions for all batches
            for validation_batch in iter(validation_loader):
                _, flattened_preds_tf = model.predict_step(
                    {"current": validation_batch.to(device)})  # .float().to(device)})    # predict using the model
                preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
            preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                                  (*batch_shape, *I.shape[-2:]))  # Combine the predictions for all batches
            # print("preds_tf.shape:", preds_tf.shape)
            preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                      axis=-2)  # concatenate the predictions
            # print("preds_tf.shape:", preds_tf.shape)
    end = time.time()  # end the timer for transformer predictions
    print("time elapsed for MOP Pred:", (end - start) / 60, "min")  # print the time elapsed for transformer predictions

    errs_tf = np.linalg.norm((ys - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    err_lss["MOP"] = errs_tf

    os.makedirs(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt", exist_ok=True)
    with open(parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl", 'wb') as f:
            pickle.dump(err_lss, f)  
    
    del errs_tf
    del preds_tf
    

    torch.cuda.empty_cache()
    gc.collect()

    irreducible_error = np.array([np.trace(sim_obj.S_observation_inf) for sim_obj in sim_objs])

    return err_lss, irreducible_error

def populate_val_traces_helper(config, trial, ys_trial, sys_choices=None, sys_dict=None, tok_seg_lens=None, real_seg_lens=None):

    # a function to populate the validation traces

    if sys_dict:
        context_len = config.n_positions + 1 #the length of the context

        segments = np.zeros((context_len, config.ny + 2*config.max_sys_trace + 2)) #initialize the segments array
        segments[0, 2*config.max_sys_trace] = np.sqrt(2) #set the start token for the first segment

        #initialize a dictionary to hold the next starting index for each system trace
        if config.late_start is not None:
            next_start_ind = config.late_start
        else:
            next_start_ind = 0

        next_start = {sys_ind: next_start_ind for sys_ind in sys_dict.keys()}
        seg_start = 1
        count = 0
        for sys in sys_choices:

            #get obs from the system trace corresponding to sys_trace_ind
            sys_trace_obs = ys_trial[sys]
            tok_seg_len = tok_seg_lens[count]
            seg_len = real_seg_lens[count]

            # Create the special tokens
            if config.needle_in_haystack and config.paren_swap and count == config.num_sys_haystack: #swap open token for query experiment

                #find the index in sys_dict.keys() where sys is located
                index_of_sys = list(sys_dict.keys()).index(sys) #get the index of the system in the sys_dict
                #swap the system index with the next system index
                swap_sys = list(sys_dict.keys())[(index_of_sys + 1) % len(sys_dict.keys())] #get the next system index as a cycle

                start_paren, end_paren = special_tokens(segment, sys_dict[swap_sys], style="zeros") #get the special tokens for the segment

            else:
                start_paren, end_paren = special_tokens(segments, sys_dict[sys], style="zeros")

            if tok_seg_len == 0: #nothing
                count += 1
                continue
            elif tok_seg_len == 1: #close parenthesis

                segments[seg_start:seg_start + tok_seg_len, :] = end_paren #add the segment to the segments array
            elif tok_seg_len == 2: #open close parenthesis

                segments[seg_start:seg_start + tok_seg_len, :] = np.concatenate([start_paren, end_paren], axis=0) #add the segment to the segments array
            else:
            
                segment = sys_trace_obs[next_start[sys]:next_start[sys] + seg_len, :] #get the segment from the next starting index to the next starting index plus the segment length

                #concatenate 1 column of ones to the segment
                ones = np.ones((segment.shape[0], 1))
                segment = np.concatenate((ones, segment), axis=1)
                #concatenate 2*max_sys_trace + 1 columns of zeros to the segment
                zeros = np.zeros((segment.shape[0], 2*config.max_sys_trace + 1))
                segment = np.concatenate((zeros, segment), axis=1)
                
                segment = np.concatenate([start_paren, segment, end_paren], axis=0)

                if seg_start + seg_len + 2 > context_len:
                    #truncate the segment if it is too long so that it fits in the context
                    segment = segment[:context_len - seg_start, :]

                segments[seg_start:seg_start + tok_seg_len, :] = segment #add the segment to the segments array
                
                next_start[sys] += seg_len #update the next starting index for the trace from this system index

            if seg_start + tok_seg_len == context_len:
                break

            seg_start += tok_seg_len #update the starting index for the next segment

            count += 1
    else:
        if trial == 0:
            raise ValueError(f"first conditional malfunction since trial = {trial}")
        else:
            raise ValueError(f"sys_dict is {sys_dict} when trial is {trial}")
    return segments, sys_choices, sys_dict, tok_seg_lens, real_seg_lens

def cycle_list(lst, shift):
    shift = shift % len(lst)
    # a function to cycle a list to the right by shift amount
    return lst[-shift:] + lst[:-shift]

def populate_val_traces(config, trace_conf, trial, num_tasks, entries, sys_choices=None, sys_dict=None, tok_seg_lens=None, seg_starts=None, real_seg_lens=None, sys_inds = None, train_conv=False, ex=None):
    # a function to populate the validation traces
    # in order to narrow the error bars, there will be num_trials versions of the same test trace configuration (randomly configured by the leader trace) with different trace realizations

    ys_trial = entries[:, trial] #get the observations for the first trial

    if trial == 0: #if this is the leader trace that sets the system indices, starting indices, and token segment lengths
        if trace_conf > 0 and config.needle_in_haystack:
            haystack = sys_choices[:-1] #get the haystack from the previous trial
            new_haystack = cycle_list(haystack, 1) #cycle the haystack to the left by 1
            sys_choices = new_haystack + [sys_choices[-1]] #set the new system choices
            segments, sys_choices, sys_dict, tok_seg_lens, real_seg_lens = populate_val_traces_helper(config, trial, ys_trial, sys_choices=sys_choices, sys_dict=sys_dict, tok_seg_lens=tok_seg_lens, real_seg_lens=real_seg_lens)

        else:
            segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(config, num_tasks, ys_trial, test=True, train_conv=train_conv, trace_conf=trace_conf, example=ex)
    else:
        segments, sys_choices, sys_dict, tok_seg_lens, real_seg_lens = populate_val_traces_helper(config, trial, ys_trial, sys_choices=sys_choices, sys_dict=sys_dict, tok_seg_lens=tok_seg_lens, real_seg_lens=real_seg_lens)
  
    return segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds

def compute_kf_multi_sys(num_trace_configs, ys, seg_lens_per_config, sys_choices_per_config, next_start_per_config, sys_inds_per_config, sim_obs_per_config, err_lss):
    
    print(f"ys shape: {ys.shape}")
    err_lss[f"Kalman_rem"] = np.full((num_trace_configs, ys.shape[1], ys.shape[2]), np.inf)

    print("computing kf multi sys with remembering")
    for trace_conf in range(num_trace_configs):
            print(f"\n\nTrace config: {trace_conf}")

            sim_objs_list = list(sim_obs_per_config[trace_conf].values())
            # print(seg_lens_per_config[trace_conf])
            #create a new array named ys_sys that takes the ys from axis 0 that correspond to every index in sys_inds_per_config[trace_conf]
            ys_sys = ys[sys_inds_per_config[trace_conf]]
            print(f"ys_sys shape: {ys_sys.shape}, len of sys_inds_per_config[trace_conf]: {len(sys_inds_per_config[trace_conf])}, ys shape: {ys.shape}")

            #think about how to take this out of the trace_conf loop
            preds_kf = np.array([[
                apply_kf(sim_obj, __ys, sigma_w=sim_obj.sigma_w,
                        sigma_v=sim_obj.sigma_v)
                for __ys in _ys
            ] for sim_obj, _ys in zip(sim_objs_list, np.take(ys_sys, np.arange(ys_sys.shape[-2] - 1), axis=-2))
            ])  # get kalman filter predictions
            print(f"preds_kf shape: {preds_kf.shape}")
            errs_kf = np.linalg.norm((ys_sys - preds_kf), axis=-1) ** 2
            print(f"errs_kf shape: {errs_kf.shape}")

            sys_start = {sys: 0 for sys in sys_inds_per_config[trace_conf]} #initialize the starting index for each system trace
            seg_count = 0
            for next_start in next_start_per_config[trace_conf]:
                sys = sys_choices_per_config[trace_conf][seg_count]

                #find the index of sys in sys_inds_per_config[trace_conf]
                sys_ind = sys_inds_per_config[trace_conf].index(sys)

                seg_len = seg_lens_per_config[trace_conf][seg_count] # get the length of the segment
                err_lss[f"Kalman_rem"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_kf[sys_ind, :, sys_start[sys]:sys_start[sys] + seg_len]
                sys_start[sys] += seg_len

                seg_count += 1
    return err_lss

def interleave_kf_OLS_needle(config, ys, errs_all, seg_lens_per_config, sys_choices_per_config, next_start_per_config, sys_inds_per_config, max_ir_length, err_lss):

    num_trace_configs = config.num_test_traces_configs
    err_lss[f"Kalman_rem"] = np.full((num_trace_configs, ys.shape[1], config.n_positions + 1), np.inf)
    err_lss[f"Analytical_Kalman"] = np.full((num_trace_configs, config.n_positions + 1), np.inf)
    err_lss[f"Analytical_Simulation"] = np.full((num_trace_configs, ys.shape[1], config.n_positions + 1), np.inf)

    for ir_length in range(1, max_ir_length + 1):
        # Initialize the err_lss ols and ols analytical values to infinity shaped like the multi_sys_ys
        err_lss[f"OLS_ir_{ir_length}"] = np.full((num_trace_configs, ys.shape[1], config.n_positions + 1), np.inf)
        err_lss[f"OLS_analytical_ir_{ir_length}"] = np.full((num_trace_configs, ys.shape[1], config.n_positions + 1), np.inf)

    for trace_conf in range(num_trace_configs):

            if config.late_start is not None:
                sys_start_ind = config.late_start
            else:
                sys_start_ind = 0

            sys_start = {sys: sys_start_ind for sys in sys_inds_per_config[trace_conf]} #initialize the starting index for each system trace
            seg_count = 0
            for next_start in next_start_per_config[trace_conf]:
                sys = sys_choices_per_config[trace_conf][seg_count]

                seg_len = seg_lens_per_config[trace_conf][seg_count] # get the length of the segment
                err_lss[f"Kalman_rem"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_all["Kalman"][sys, :, sys_start[sys]:sys_start[sys] + seg_len]
                err_lss[f"Analytical_Kalman"][trace_conf, next_start + 1:next_start + 1 + seg_len] = errs_all["Analytical_Kalman"][sys, sys_start[sys]:sys_start[sys] + seg_len]
                err_lss[f"Analytical_Simulation"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_all["Analytical_Simulation"][sys, :, sys_start[sys]:sys_start[sys] + seg_len]

                for ir_length in range(1, max_ir_length + 1):
                    err_lss[f"OLS_ir_{ir_length}"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_all[f"OLS_ir_{ir_length}"][sys, :, sys_start[sys]:sys_start[sys] + seg_len]
                    err_lss[f"OLS_analytical_ir_{ir_length}"][trace_conf, :, next_start + 1:next_start + 1 + seg_len] = errs_all[f"OLS_analytical_ir_{ir_length}"][sys, :, sys_start[sys]:sys_start[sys] + seg_len]

                sys_start[sys] += seg_len

                seg_count += 1

    return err_lss

def compute_errors_multi_sys(config, tf, run_OLS=True, train_conv=False, run_kf=True):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available
    logger = logging.getLogger(__name__)  # get the logger

    num_systems = config.num_val_tasks  # number of validation tasks
    
    if ((not config.needle_in_haystack) or config.datasource == "val" or config.datasource == "train_systems"):
        num_trials = config.num_traces["val"]
    elif config.datasource == "train":
        num_trials = config.num_traces["train"]
    else:
        raise ValueError(f"datasource {config.datasource} not recognized")
    
    num_test_traces_configs = config.num_test_traces_configs

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                      n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                      n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                      n_layer=config.n_layer, n_head=config.n_head, map_location=device, use_pos_embd=config.use_pos_embd).eval().to(
        device)  # load_from_checkpoint

    
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    ckpt_steps = get_step_number(config.ckpt_path)
    
    #create a directory to save the prediction errors
    errs_dir = parent_parent_dir + f"/prediction_errors" + ("_spec_C" if config.needle_in_haystack and config.datasource == "train_systems" and config.multi_sys_trace else f"{config.C_dist}") + f"_step={ckpt_steps}.ckpt"
    errs_loc = errs_dir + f"/" + ("train_conv_" if train_conv else "") + ("single_system_" if config.single_system else "") + ("zero_cut_" if config.zero_cut else "") + (f"needle_haystack_len_{config.num_sys_haystack}_{config.datasource}_" if config.needle_in_haystack else "") + f"{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl"

    if os.path.exists(errs_loc):
        with open(errs_loc, 'rb') as f:
            err_lss = pickle.load(f)

        print(f"\n err_lss already exists for step {ckpt_steps}")

    else:
        err_lss = collections.OrderedDict()
        print(f"\n err_lss does not exist yet for step {ckpt_steps}")


    # print("no tf pred")
    # Transformer Predictions
    # if not ("MOP" in err_lss.keys()):


    multi_sys_ys = np.zeros((num_test_traces_configs, num_trials, config.n_positions + 1, config.ny + 2*config.max_sys_trace + 2)).astype(np.float32) #set up the array to hold the test traces

    #get the ys and sim_objs for the test data 
    if ((not config.needle_in_haystack) or config.datasource == "val"):

        print(f"getting test data from datasource {config.datasource}")

        # get the sim objs for the validation data
        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        #set ys to be the validation data
        with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
            samples = pickle.load(f)
            # for every 2000 entries in samples, get the observation values and append them to the ys list
            ys = np.stack(
                [entry["obs"] for entry in samples], axis=0
            ).reshape((num_systems, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)

            gc.collect()  # Start the garbage collector

    elif config.datasource == "train":

        print(f"getting test data from datasource {config.datasource}")

        #get the sim_objs for the training data
        with open (parent_parent_dir + f"/data/train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        #set ys to be the training data
        with open(parent_parent_dir + f"/data/train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
            #get train traces
            samples = pickle.load(f)
            ys = np.stack(
                [entry["obs"] for entry in samples], axis=0
            ).reshape((config.num_tasks, config.num_traces["train"], config.n_positions + 1, config.ny)).astype(np.float32)
            gc.collect()  # Start the garbage collector

    elif config.datasource == "train_systems":

        print(f"getting test data from datasource {config.datasource}")

        #get the sim_objs for the training data
        with open(parent_parent_dir + f"/data/train_{config.dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        #generate traces from the training systems
        collect_data(config, parent_parent_dir, "val", False, False, False, sim_objs) 

        with open(parent_parent_dir + f"/data/{config.datasource}_val_specA_spec_C_state_dim_{config.nx}.pkl", "rb") as f:
            #get train traces
            samples = pickle.load(f)
            ys = np.stack(
                [entry["obs"] for entry in samples], axis=0
            ).reshape((config.num_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)
            gc.collect()  # Start the garbage collector

    else:
        raise ValueError(f"datasource {config.datasource} not recognized")
        

    sys_choices_per_config = []
    sys_dict_per_config = []
    tok_seg_lens_per_config = []
    seg_starts_per_config = []
    real_seg_lens_per_config = []
    sys_inds_per_config = []

    print("\n\nstart populating test traces")
    for trace_config in range(num_test_traces_configs):
        print(f"\nTrace config: {trace_config}\n")
        #ys are of dim: (num_systems, num_trials, config.n_positions + 1, config.ny)
        if (not config.needle_in_haystack) or (trace_config == 0):
            tok_seg_lens = None
            sys_dict = None
            sys_choices = None
            seg_starts= None
            real_seg_lens=None
            sys_inds = None
        for trial in range(num_trials):

            #generate interleaved segments
            segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_val_traces(config, trace_config, trial, config.num_val_tasks, ys, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds, train_conv) # get the first trace  which will set the testing structure
            multi_sys_ys[trace_config, trial] = segments
        
        sys_choices_per_config.append(sys_choices)
        sys_dict_per_config.append(sys_dict)
        tok_seg_lens_per_config.append(tok_seg_lens)
        seg_starts_per_config.append(seg_starts)
        real_seg_lens_per_config.append(real_seg_lens)
        sys_inds_per_config.append(sys_inds)

    print("\nstart tf pred")
    start = time.time()  # start the timer for transformer predictions
    with torch.no_grad():  # no gradients
        I = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)

        # print("before model.predict_step()")
        batch_shape = I.shape[:-2]
        # print("batch_shape:", batch_shape)
        flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))
        # print("flattened_I.shape:", flattened_I.shape)
        validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I),
                                                        batch_size=config.test_batch_size)
        preds_arr = []  # Store the predictions for all batches
        for validation_batch in iter(validation_loader):
            _, flattened_preds_tf = model.predict_step(
                {"current": validation_batch.to(device)})  # .float().to(device)})    # predict using the model
            preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
        preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                            (*batch_shape, config.n_positions, config.ny))  # Combine the predictions for all batches
        # print("preds_tf.shape:", preds_tf.shape)
        preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                axis=-2)  # concatenate the predictions
        # print("preds_tf.shape:", preds_tf.shape)
    end = time.time()  # end the timer for transformer predictions
    print("time elapsed for MOP Pred:", (end - start) / 60, "min")  # print the time elapsed for transformer predictions

    #take the last config.ny columns of axis=-1 as the true test observations
    multi_sys_ys_true = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-1] - config.ny, multi_sys_ys.shape[-1]), axis=-1) #get the true test observations

    errs_tf = np.linalg.norm((multi_sys_ys_true - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    for trace_config in range(num_test_traces_configs):
        #set the errors for the start token to be infinite
        errs_tf[trace_config, :, 0] = np.inf
        errs_tf[trace_config, :, 1] = np.inf
        seg_count = 0
        for seg_start in seg_starts_per_config[trace_config]: #loop over the starting indices of the segments
            #set the errors of the end of the segment to be infinite
            if real_seg_lens_per_config[trace_config][seg_count] < tok_seg_lens_per_config[trace_config][seg_count] - 1:
                errs_tf[trace_config, :, seg_start + tok_seg_lens_per_config[trace_config][seg_count] - 1] = np.inf
            if seg_start + tok_seg_lens_per_config[trace_config][seg_count] <= config.n_positions:
                errs_tf[trace_config, :, seg_start + tok_seg_lens_per_config[trace_config][seg_count]] = np.inf
            seg_count += 1


    err_lss["MOP"] = errs_tf


    os.makedirs(errs_dir, exist_ok=True)
    with open(errs_loc, 'wb') as f:
            pickle.dump(err_lss, f)  
    
    del errs_tf
    del preds_tf

    torch.cuda.empty_cache()
    gc.collect()

    if tf and not (run_kf or run_OLS): #only run transformer predictions
        return err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config

    print("start zero predictor")
    # zero predictor predictions
    errs_zero = np.linalg.norm(multi_sys_ys_true, axis=-1) ** 2  # get the errors of zero predictions
    err_lss["Zero"] = errs_zero

    if config.num_haystack_examples == 1:
        for i in range(len(sys_choices_per_config)):
            print(f"trace config: {i}")
            print(f"len of sys_choices: {len(sys_choices_per_config[i])}")
            print("sum of zero err:", np.sum(errs_zero[i]))

    os.makedirs(errs_dir, exist_ok=True)
    with open(errs_loc, 'wb') as f:
            pickle.dump(err_lss, f)

    del errs_zero

    torch.cuda.empty_cache()
    gc.collect()


    if True:
        start = time.time()  # start the timer for kalman filter predictions
        
        #create a list of sim_objs for each trace configuration by accessing the sim_objs using the system indices
        sim_objs_per_config = []
        for trace_config in range(num_test_traces_configs):
            sim_obj_conf = {}
            for sys_ind in sys_inds_per_config[trace_config]:
                sim_obj_conf[sys_ind] = sim_objs[sys_ind] 

            sim_objs_per_config.append(sim_obj_conf)


        # print("no kf pred")
        # Kalman Predictions
        print("start kf pred")
        preds_kf = np.zeros(multi_sys_ys_true.shape)
        an_kf_errs = np.zeros((num_test_traces_configs, config.n_positions + 1)) #initialize the array to hold the analytical kalman filter errors
        an_sim_preds = np.zeros((num_test_traces_configs, num_trials, config.n_positions + 1, config.ny)) #initialize the array to hold the analytical simulation predictions
        conf_count = 0
        for sim_obj_dict_conf, _ys in zip(sim_objs_per_config, np.take(multi_sys_ys_true, np.arange(multi_sys_ys_true.shape[-2]), axis=-2)): #loop over trace_configuration
            
            seg_starts_conf = seg_starts_per_config[conf_count]
            # start_inds_conf = start_inds_per_config[conf_count] #get the starting indices for the trace configuration
            tok_seg_lens_conf = tok_seg_lens_per_config[conf_count]
            sys_choices_conf = sys_choices_per_config[conf_count]
            real_seg_lens_conf = real_seg_lens_per_config[conf_count]

            inner_result = np.zeros(multi_sys_ys_true[0].shape) # initialize the kf pred array holder for this trace configuration
            inner_result[:, 0, :] = np.inf #set the kalman prediction error for start token to be infinite
            trial_count = 0
            for __ys in _ys: #loop over trial in trace configuration

                # la.print_matrix(__ys, "ys")
                seg_count = 0 #count of which segment
                for seg_start in seg_starts_conf: #loop over the starting indices of the segments
                    sim_obj = sim_objs[sys_choices_conf[seg_count]]

                    #set the kalman prediction error for start paren to be infinite
                    inner_result[trial_count, seg_start, :] = np.inf 
                    #set the kalman prediction error for end paren to be infinite
                    inner_result[trial_count, seg_start + tok_seg_lens_conf[seg_count]-1, :] = np.inf 

                    #get the observation values for the segment of ys without the special tokens
                    ys_seg = __ys[seg_start + 1:seg_start + 1 + real_seg_lens_conf[seg_count], :]


                    # la.print_matrix(ys_seg, "ys_seg")
                    if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho"):
                        # Apply the Kalman filter and append the result to the inner list
                        result = apply_kf(sim_obj, ys_seg, sigma_w=sim_obj.sigma_w, sigma_v=sim_obj.sigma_v)

                        #remove the last kf pred because the true y was a special token, and insert the kf pred after the last kf preds
                        inner_result[trial_count, seg_start + 1:seg_start + 1 + real_seg_lens_conf[seg_count], :] = result[:-1,:] #if last index was not an end_paren the infinity will be overwritten here

                    #analytical kalman filter errors
                    #get the analytical kalman filter error for the segment
                    an_kf_errs[conf_count, seg_start:seg_start + tok_seg_lens_conf[seg_count]] = np.trace(sim_obj.S_observation_inf) * np.ones(tok_seg_lens_conf[seg_count]) 

                    #analytical simulation predicions
                    #get the analytical simulation predictions for the segment
                    an_sim_preds[conf_count, trial_count, seg_start:seg_start + tok_seg_lens_conf[seg_count], :] = np.random.multivariate_normal(np.zeros(config.ny), sim_obj.S_observation_inf, (tok_seg_lens_conf[seg_count])) 

                    seg_count += 1
                trial_count += 1

            # Append the inner list to the preds_kf list
            preds_kf[conf_count] = inner_result
            conf_count += 1

        if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho"):
            # Convert the preds_kf list to a numpy array
            preds_kf = np.array(preds_kf)
            errs_kf = np.linalg.norm((multi_sys_ys_true - preds_kf), axis=-1) ** 2
            err_lss["Kalman"] = errs_kf

            os.makedirs(errs_dir, exist_ok=True)
            with open(errs_loc, 'wb') as f:
                    pickle.dump(err_lss, f)

            end = time.time()  # end the timer for kalman filter predictions
            print("time elapsed for KF Pred:", (end - start) / 60,
                "min")  # print the time elapsed for kalman filter predictions

            del preds_kf
            del errs_kf

            torch.cuda.empty_cache()
            gc.collect()

    else:
        print("Kalman pred for ident is trivial")

    
    err_lss["Analytical_Kalman"] = an_kf_errs #set the analytical kalman filter errors in the err_lss dictionary
    err_lss["Analytical_Simulation"] = np.linalg.norm(an_sim_preds, axis=-1) ** 2 #set the analytical simulation predictions in the err_lss dictionary

    os.makedirs(errs_dir, exist_ok=True)
    with open(errs_loc, 'wb') as f:
            pickle.dump(err_lss, f)

    if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho"):
        print("kf multi sys with remembering")
        start = time.time()  # start the timer for kalman filter predictions
        err_lss = compute_kf_multi_sys(num_test_traces_configs, ys, real_seg_lens_per_config, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, sim_objs_per_config, err_lss)
        end = time.time()  # end the timer for kalman filter predictions
        print("time elapsed for KF Pred with remembering:", (end - start) / 60, "min")  # print the time elapsed for kalman filter predictions with remembering

        os.makedirs(errs_dir, exist_ok=True)
        with open(errs_loc, 'wb') as f:
                pickle.dump(err_lss, f)


    # Think about having a sys_choices to prediction error dictionary to implement the remembering
    if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho") and run_OLS:
    # if not ("OLS" in err_lss.keys()):
        # Original OLS
        # Clear the PyTorch cache
        start = time.time()  # start the timer for OLS predictions
        print("start OLS pred")

        err_lss = compute_OLS_ir_multi_sys(num_test_traces_configs, seg_starts_per_config, real_seg_lens_per_config, sys_choices_per_config, sys_inds_per_config, sim_objs_per_config, config, ys, max_ir_length=3, err_lss=err_lss)


        os.makedirs(errs_dir, exist_ok=True)
        with open(errs_loc, 'wb') as f:
                pickle.dump(err_lss, f)

        end = time.time()  # end the timer for OLS predictions
        print("time elapsed for OLS Pred:", (end - start) / 60, "min")  # print the time elapsed for OLS predictions
    else:
        print("OLS pred for ident is trivial")

    return err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config


def compute_errors_needle(config, model, ys, sim_objs, errs_dir, errs_loc, ex=None):
    # a function to compute the test errors for the GPT2 model, kalman filter, and zero predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    num_systems = config.num_val_tasks  # number of validation tasks
    
    if ((not config.needle_in_haystack) or config.datasource == "val" or config.datasource == "train_systems"):
        num_trials = config.num_traces["val"]
    elif config.datasource == "train":
        num_trials = config.num_traces["train"]
    else:
        raise ValueError(f"datasource {config.datasource} not recognized")
    
    num_test_traces_configs = config.num_test_traces_configs

    
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    ckpt_steps = get_step_number(config.ckpt_path)


    # if os.path.exists(errs_loc):
    #     with open(errs_loc, 'rb') as f:
    #         err_lss = pickle.load(f)

    #     print(f"\n err_lss already exists for step {ckpt_steps}")

    # else:
    err_lss = collections.OrderedDict()
    # print(f"\n err_lss does not exist yet for step {ckpt_steps}")


    multi_sys_ys = np.zeros((num_test_traces_configs, num_trials, config.n_positions + 1, config.ny + 2*config.max_sys_trace + 2)).astype(np.float32) #set up the array to hold the test traces
        

    sys_choices_per_config = []
    sys_dict_per_config = []
    tok_seg_lens_per_config = []
    seg_starts_per_config = []
    real_seg_lens_per_config = []
    sys_inds_per_config = []

    # start = time.time()  # start the timer for transformer predictions
    for trace_config in range(num_test_traces_configs):
        #ys are of dim: (num_systems, num_trials, config.n_positions + 1, config.ny)
        if (not config.needle_in_haystack) or (trace_config == 0):
            tok_seg_lens = None
            sys_dict = None
            sys_choices = None
            seg_starts= None
            real_seg_lens=None
            sys_inds = None
        for trial in range(num_trials):

            #generate interleaved segments
            segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_val_traces(config, trace_config, trial, config.num_val_tasks, ys, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds, ex=ex) # get the first trace  which will set the testing structure
            multi_sys_ys[trace_config, trial] = segments
        
        sys_choices_per_config.append(sys_choices)
        sys_dict_per_config.append(sys_dict)
        tok_seg_lens_per_config.append(tok_seg_lens)
        seg_starts_per_config.append(seg_starts)
        real_seg_lens_per_config.append(real_seg_lens)
        sys_inds_per_config.append(sys_inds)

    # end = time.time()  # end the timer for transformer predictions
    # print("time elapsed for populating test traces:", (end - start), "sec")  # print the time elapsed for populating the test traces

    # print("\nstart tf pred")
    # start = time.time()  # start the timer for transformer predictions
    with torch.no_grad():  # no gradients
        I = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)

        # print("before model.predict_step()")
        batch_shape = I.shape[:-2]
        # print("batch_shape:", batch_shape)
        flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))
        # print("flattened_I.shape:", flattened_I.shape)
        validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I),
                                                        batch_size=config.test_batch_size)
        preds_arr = []  # Store the predictions for all batches
        for validation_batch in iter(validation_loader):
            _, flattened_preds_tf = model.predict_step(
                {"current": validation_batch.to(device)})  # .float().to(device)})    # predict using the model
            preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
        preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                            (*batch_shape, config.n_positions, config.ny))  # Combine the predictions for all batches
        # print("preds_tf.shape:", preds_tf.shape)
        preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf],
                                axis=-2)  # concatenate the predictions
        # print("preds_tf.shape:", preds_tf.shape)
    # end = time.time()  # end the timer for transformer predictions
    # print("time elapsed for MOP Pred:", (end - start), "sec")  # print the time elapsed for transformer predictions

    #take the last config.ny columns of axis=-1 as the true test observations
    multi_sys_ys_true = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-1] - config.ny, multi_sys_ys.shape[-1]), axis=-1) #get the true test observations

    errs_tf = np.linalg.norm((multi_sys_ys_true - preds_tf), axis=-1) ** 2  # get the errors of transformer predictions
    for trace_config in range(num_test_traces_configs):
        #set the errors for the start token to be infinite
        errs_tf[trace_config, :, 0] = np.inf
        errs_tf[trace_config, :, 1] = np.inf
        seg_count = 0
        for seg_start in seg_starts_per_config[trace_config]: #loop over the starting indices of the segments
            #set the errors of the end of the segment to be infinite
            if real_seg_lens_per_config[trace_config][seg_count] < tok_seg_lens_per_config[trace_config][seg_count] - 1:
                errs_tf[trace_config, :, seg_start + tok_seg_lens_per_config[trace_config][seg_count] - 1] = np.inf
            if seg_start + tok_seg_lens_per_config[trace_config][seg_count] <= config.n_positions:
                errs_tf[trace_config, :, seg_start + tok_seg_lens_per_config[trace_config][seg_count]] = np.inf
            seg_count += 1


    err_lss["MOP"] = errs_tf


    os.makedirs(errs_dir, exist_ok=True)
    with open(errs_loc, 'wb') as f:
            pickle.dump(err_lss, f)  
    
    del errs_tf
    del preds_tf

    torch.cuda.empty_cache()
    gc.collect()

    # print("start zero predictor")
    # zero predictor predictions
    errs_zero = np.linalg.norm(multi_sys_ys_true, axis=-1) ** 2  # get the errors of zero predictions
    err_lss["Zero"] = errs_zero

    if config.num_haystack_examples == 1:
        for i in range(len(sys_choices_per_config)):
            print(f"trace config: {i}")
            print(f"len of sys_choices: {len(sys_choices_per_config[i])}")
            print("sum of zero err:", np.sum(errs_zero[i]))

    os.makedirs(errs_dir, exist_ok=True)
    with open(errs_loc, 'wb') as f:
            pickle.dump(err_lss, f)

    del errs_zero

    torch.cuda.empty_cache()
    gc.collect() 

    #create a list of sim_objs for each trace configuration by accessing the sim_objs using the system indices
    sim_objs_per_config = []
    for trace_config in range(num_test_traces_configs):
        sim_obj_conf = {}
        for sys_ind in sys_inds_per_config[trace_config]:
            sim_obj_conf[sys_ind] = sim_objs[sys_ind] 

        sim_objs_per_config.append(sim_obj_conf)  

    return err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, real_seg_lens_per_config, sys_inds_per_config, sim_objs_per_config

def interleave_traces(config, ys, num_test_traces_configs, num_trials, ex=None):
    multi_sys_ys = np.zeros((num_test_traces_configs, num_trials, config.n_positions + 1, config.ny + 2*config.max_sys_trace + 2)).astype(np.float32) #set up the array to hold the test traces
        

    sys_choices_per_config = []
    sys_dict_per_config = []
    tok_seg_lens_per_config = []
    seg_starts_per_config = []
    real_seg_lens_per_config = []
    sys_inds_per_config = []

    # start = time.time()  # start the timer for transformer predictions
    for trace_config in range(num_test_traces_configs):
        #ys are of dim: (num_systems, num_trials, config.n_positions + 1, config.ny)
        if (not config.needle_in_haystack) or (trace_config == 0):
            tok_seg_lens = None
            sys_dict = None
            sys_choices = None
            seg_starts= None
            real_seg_lens=None
            sys_inds = None
        for trial in range(num_trials):

            #generate interleaved segments
            segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds = populate_val_traces(config, trace_config, trial, config.num_val_tasks, ys, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds, ex=ex) # get the first trace  which will set the testing structure
            multi_sys_ys[trace_config, trial] = segments
        
        sys_choices_per_config.append(sys_choices)
        sys_dict_per_config.append(sys_dict)
        tok_seg_lens_per_config.append(tok_seg_lens)
        seg_starts_per_config.append(seg_starts)
        real_seg_lens_per_config.append(real_seg_lens)
        sys_inds_per_config.append(sys_inds)

    return multi_sys_ys, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, real_seg_lens_per_config, sys_inds_per_config

def needle_in_haystack_preds(config, model, ckpt_steps, parent_parent_dir, errs_dir, train_conv, ys, sim_objs, run_kf_ols=True):

    print(f"config.num_haystack_examples: {config.num_haystack_examples}")

    save_errs_dir = parent_parent_dir + f"/prediction_errors" + ("_spec_C" if config.needle_in_haystack and config.datasource == "train_systems" and config.multi_sys_trace else f"{config.C_dist}") + f"_step={ckpt_steps}.ckpt"
    save_errs_loc = errs_dir + f"/" + ("single_system_" if config.single_system else "") + ("train_conv_" if train_conv else "") + (f"needle_haystack_len_{config.num_sys_haystack}_{config.datasource}_" if config.needle_in_haystack else "") + ("fin_seg_ext_" if config.needle_in_haystack and config.needle_final_seg_extended else "") + f"{config.val_dataset_typ}_state_dim_{config.nx}_"+ ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("paren_swap_" if config.paren_swap else "")
    # if (config.datasource == "val"):

    #     print(f"getting test data from datasource {config.datasource}")

    #     # get the sim objs for the validation data
    #     with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
    #         sim_objs = pickle.load(f)

    #     #set ys to be the validation data
    #     with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
    #         samples = pickle.load(f)
    #         # for every 2000 entries in samples, get the observation values and append them to the ys list
    #         ys = np.stack(
    #             [entry["obs"][:config.n_positions + 1] for entry in samples], axis=0
    #         ).reshape((config.num_val_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)

    #         gc.collect()  # Start the garbage collector


    #     #now that I have err_lss_all, I just need to interleave everything instead of computing for each example and trace configuration

    # elif config.datasource == "train":

    #     print(f"getting test data from datasource {config.datasource}")

    #     #get the sim_objs for the training data
    #     with open (parent_parent_dir + f"/data/train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
    #         sim_objs = pickle.load(f)

    #     #set ys to be the training data
    #     with open(parent_parent_dir + f"/data/train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
    #         #get train traces
    #         samples = pickle.load(f)
    #         ys = np.stack(
    #             [entry["obs"] for entry in samples], axis=0
    #         ).reshape((config.num_tasks, config.num_traces["train"], config.n_positions + 1, config.ny)).astype(np.float32)
    #         gc.collect()  # Start the garbage collector

    # elif config.datasource == "train_systems":

    #     print(f"getting test data from datasource {config.datasource}")

    #     #get the sim_objs for the training data
    #     with open(parent_parent_dir + f"/data/train_{config.dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
    #         sim_objs = pickle.load(f)

    #     #generate traces from the training systems
    #     collect_data(config, parent_parent_dir, "val", False, False, False, sim_objs) 

    #     with open(parent_parent_dir + f"/data/{config.datasource}_val_specA_spec_C_state_dim_{config.nx}.pkl", "rb") as f:
    #         #get train traces
    #         samples = pickle.load(f)
    #         ys = np.stack(
    #             [entry["obs"] for entry in samples], axis=0
    #         ).reshape((config.num_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)
    #         gc.collect()  # Start the garbage collector

    # else:
    #     raise ValueError(f"datasource {config.datasource} not recognized")
    

    err_lss_all = {}

    if (not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho")) and run_kf_ols:

        if ((not config.needle_in_haystack) or config.datasource == "val" or config.datasource == "train_systems"):
            num_trials = config.num_traces["val"]
        elif config.datasource == "train":
            num_trials = config.num_traces["train"]
        else:
            raise ValueError(f"datasource {config.datasource} not recognized")
    
        start = time.time()  # start the timer for kf predictions
        errs_kf = compute_kf(ys, sim_objs)
        end = time.time()  # end the timer for kf predictions
        print("time elapsed for KF Pred:", (end - start) / 60, "min")  # print the time elapsed for kf predictions
        err_lss_all["Kalman"] = errs_kf

        start = time.time()  # start the timer for ols predictions
        err_lss_all = compute_OLS_ir(config, ys, sim_objs, max_ir_length=3, err_lss=err_lss_all)
        end = time.time()
        print("time elapsed for OLS Pred:", (end - start) / 60, "min")  # print the time elapsed for OLS predictions

        # #for quick debugging
        # names = ["Kalman", "Kalman_rem", "OLS_ir_1", "OLS_ir_2", "OLS_ir_3"]
        # for name in names:
        #     err_lss_all[name] = np.zeros((config.num_val_tasks, num_trials, config.n_positions + 1))

        analytical_kf, an_sims = compute_analytical_kf_simulation(config, ys, sim_objs, num_trials)
        err_lss_all["Analytical_Kalman"] = analytical_kf
        print(f"err_lss_all[Analytical_Kalman].shape: {err_lss_all['Analytical_Kalman'].shape}")
        err_lss_all["Analytical_Simulation"] = an_sims
        print(f"err_lss_all[Analytical_Simulation].shape: {err_lss_all['Analytical_Simulation'].shape}")

        # raise ValueError("Need to implement interleaving of KF and OLS errors")

    err_lss_examples = {}
    for ex in range(config.num_haystack_examples):
        # start = time.time()  # start the timer for needle predictions
        err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, real_seg_lens_per_config, sys_inds_per_config, sim_objs_per_config  = compute_errors_needle(config, model, ys, sim_objs, save_errs_dir, save_errs_loc + "err_lss.pkl", ex=ex)
        end = time.time()  # end the timer for needle predictions
        # print(f"time elapsed for tf needle predictions example {ex}:", (end - start), "sec")  # print the time elapsed for needle predictions

        if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho") and run_kf_ols:

            print("interleaving kf and OLS errors")
            err_lss = interleave_kf_OLS_needle(config, ys, err_lss_all, real_seg_lens_per_config, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, max_ir_length=3, err_lss=err_lss)

        for key in err_lss.keys():
            # print(f"err_lss[{key}] len: {len(err_lss[key])}")
            if ex == 0:
                err_lss_examples[key] = [] #initialize the list for the prediction errors
            err_lss_examples[key].append(err_lss[key])


        del err_lss
        torch.cuda.empty_cache()
        gc.collect()
        
        if ex == 0:
            #save the system indices, starting indices, and token segment lengths to pickle file
            with open(save_errs_loc + f"sys_choices_sys_dict_tok_seg_lens_seg_starts_example_{ex}.pkl", 'wb') as f:
                pickle.dump({
                    'sys_choices_per_config': sys_choices_per_config,
                    'sys_dict_per_config': sys_dict_per_config,
                    'tok_seg_lens_per_config': tok_seg_lens_per_config,
                    'seg_starts_per_config': seg_starts_per_config
                }, f)
        
        end = time.time()  # end the timer for needle predictions
        # print(f"time elapsed for Needle Pred example {ex}:", (end - start), "sec\n\n\n")  # print the time elapsed for needle predictions

    for key in err_lss_examples.keys():
        # print(f"err_lss_examples[{key}] len: {len(err_lss_examples[key])}")
        err_lss_examples[key] = np.array(err_lss_examples[key])
        # print(f"err_lss_examples[{key}] shape: {err_lss_examples[key].shape}")

    with open(save_errs_loc + "err_lss_examples.pkl", 'wb') as f:
        pickle.dump(err_lss_examples, f)

    return None

def save_preds(run_deg_kf_test, config, model, train_conv, tf, ys, sim_objs, run_kf_ols=True):
    # make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    ckpt_steps = get_step_number(config.ckpt_path)
    print("ckpt_steps:", ckpt_steps)

    errs_dir = parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt"
    errs_loc = errs_dir + f"/" + ("single_system_" if config.single_system else "") + ("zero_cut_" if config.zero_cut else "") + (f"needle_haystack_len_{config.num_sys_haystack}_{config.datasource}_" if config.needle_in_haystack else "") + ("fin_seg_ext_" if config.needle_in_haystack and config.needle_final_seg_extended else "") + f"{config.val_dataset_typ}_state_dim_{config.nx}_"

    os.makedirs(errs_dir, exist_ok=True)

    if train_conv and not config.multi_sys_trace:
        err_lss, irreducible_error = compute_errors_conv(config)
    elif train_conv and config.multi_sys_trace:

        if not config.needle_in_haystack:
            print(f"in train conv and multi sys trace")
            print(f"config.single_system: {config.single_system}")
            print(f"config.needle_in_haystack: {config.needle_in_haystack}")

            run_OLS = run_kf_ols
            run_kf = run_kf_ols

            err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config = compute_errors_multi_sys(config, tf, run_OLS=run_OLS, train_conv=train_conv, run_kf=run_kf)

            #save the system indices, starting indices, and token segment lengths to pickle file
            with open(errs_loc + "sys_choices_sys_dict_tok_seg_lens_seg_starts.pkl", 'wb') as f:
                pickle.dump({
                    'sys_choices_per_config': sys_choices_per_config,
                    'sys_dict_per_config': sys_dict_per_config,
                    'tok_seg_lens_per_config': tok_seg_lens_per_config,
                    'seg_starts_per_config': seg_starts_per_config
                }, f)
            return None
        else:

            needle_in_haystack_preds(config, model, ckpt_steps, parent_parent_dir, errs_dir, train_conv, ys, sim_objs, run_kf_ols=run_kf_ols)
            return None



    elif not train_conv and config.multi_sys_trace:
        if not config.needle_in_haystack:
            err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config = compute_errors_multi_sys(config, tf)
            
            #save the system indices, starting indices, and token segment lengths to pickle file
            with open(errs_loc + "sys_choices_sys_dict_tok_seg_lens_seg_starts.pkl", 'wb') as f:
                pickle.dump({
                    'sys_choices_per_config': sys_choices_per_config,
                    'sys_dict_per_config': sys_dict_per_config,
                    'tok_seg_lens_per_config': tok_seg_lens_per_config,
                    'seg_starts_per_config': seg_starts_per_config
                }, f)
            return None
        else:

            # save_errs_dir = parent_parent_dir + f"/prediction_errors" + ("_spec_C" if config.needle_in_haystack and config.datasource == "train_systems" and config.multi_sys_trace else f"{config.C_dist}") + f"_step={ckpt_steps}.ckpt"
            # save_errs_loc = errs_dir + f"/" + ("single_system_" if config.single_system else "") + (f"needle_{config.datasource}_" if config.needle_in_haystack else "") + ("fin_seg_ext_" if config.needle_in_haystack and config.needle_final_seg_extended else "") + f"{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl"

            # if (config.datasource == "val"):

            #     print(f"getting test data from datasource {config.datasource}")

            #     # get the sim objs for the validation data
            #     with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            #         sim_objs = pickle.load(f)

            #     #set ys to be the validation data
            #     with open(parent_parent_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
            #         samples = pickle.load(f)
            #         # for every 2000 entries in samples, get the observation values and append them to the ys list
            #         ys = np.stack(
            #             [entry["obs"] for entry in samples], axis=0
            #         ).reshape((config.num_val_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)

            #         gc.collect()  # Start the garbage collector


            #     #now that I have err_lss_all, I just need to interleave everything instead of computing for each example and trace configuration

            # elif config.datasource == "train":

            #     print(f"getting test data from datasource {config.datasource}")

            #     #get the sim_objs for the training data
            #     with open (parent_parent_dir + f"/data/train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            #         sim_objs = pickle.load(f)

            #     #set ys to be the training data
            #     with open(parent_parent_dir + f"/data/train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
            #         #get train traces
            #         samples = pickle.load(f)
            #         ys = np.stack(
            #             [entry["obs"] for entry in samples], axis=0
            #         ).reshape((config.num_tasks, config.num_traces["train"], config.n_positions + 1, config.ny)).astype(np.float32)
            #         gc.collect()  # Start the garbage collector

            # elif config.datasource == "train_systems":

            #     print(f"getting test data from datasource {config.datasource}")

            #     #get the sim_objs for the training data
            #     with open(parent_parent_dir + f"/data/train_{config.dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            #         sim_objs = pickle.load(f)

            #     #generate traces from the training systems
            #     collect_data(config, parent_parent_dir, "val", False, False, False, sim_objs) 

            #     with open(parent_parent_dir + f"/data/{config.datasource}_val_specA_spec_C_state_dim_{config.nx}.pkl", "rb") as f:
            #         #get train traces
            #         samples = pickle.load(f)
            #         ys = np.stack(
            #             [entry["obs"] for entry in samples], axis=0
            #         ).reshape((config.num_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)
            #         gc.collect()  # Start the garbage collector

            # else:
            #     raise ValueError(f"datasource {config.datasource} not recognized")
            

            needle_in_haystack_preds(config, model, ckpt_steps, parent_parent_dir, errs_dir, train_conv, ys, sim_objs)
            return None

            # err_lss_all = {}

            # if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho"):
            
            #     start = time.time()  # start the timer for kf predictions
            #     errs_kf = compute_kf(config, ys, sim_objs)
            #     end = time.time()  # end the timer for kf predictions
            #     print("time elapsed for KF Pred:", (end - start) / 60, "min")  # print the time elapsed for kf predictions
            #     err_lss_all["Kalman"] = errs_kf

            #     start = time.time()  # start the timer for ols predictions
            #     err_lss_all = compute_OLS_ir(config, ys, sim_objs, max_ir_length=3, err_lss=err_lss_all)
            #     end = time.time()
            #     print("time elapsed for OLS Pred:", (end - start) / 60, "min")  # print the time elapsed for OLS predictions

            #     # #for quick debugging
            #     # names = ["Kalman", "Kalman_rem", "OLS_ir_1", "OLS_ir_2", "OLS_ir_3"]
            #     # for name in names:
            #     #     err_lss_all[name] = np.zeros((config.num_val_tasks, config.num_traces["val"], config.n_positions + 1))

            #     analytical_kf, an_sims = compute_analytical_kf_simulation(config, ys, sim_objs)
            #     err_lss_all["Analytical_Kalman"] = analytical_kf
            #     err_lss_all["Analytical_Simulation"] = an_sims

            # err_lss_examples = {}
            # for ex in range(config.num_haystack_examples):
            #     start = time.time()  # start the timer for needle predictions
            #     err_lss, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, real_seg_lens_per_config, sys_inds_per_config, sim_objs_per_config  = compute_errors_needle(config, ys, sim_objs, save_errs_dir, save_errs_loc)

            #     if not (config.val_dataset_typ == "ident" or config.val_dataset_typ == "ortho"):
            #         err_lss = interleave_kf_OLS_needle(config.num_test_traces_configs, ys, err_lss_all, real_seg_lens_per_config, sys_choices_per_config, seg_starts_per_config, sys_inds_per_config, max_ir_length=3, err_lss=err_lss)

            #     os.makedirs(save_errs_dir, exist_ok=True)
            #     with open(save_errs_loc, 'wb') as f:
            #             pickle.dump(err_lss, f)

            #     for key in err_lss.keys():
            #         print(f"err_lss[{key}] len: {len(err_lss[key])}")
            #         if ex == 0:
            #             err_lss_examples[key] = [] #initialize the list for the prediction errors
            #         err_lss_examples[key].append(err_lss[key])


            #     del err_lss
            #     torch.cuda.empty_cache()
            #     gc.collect()
            
            #     #save the system indices, starting indices, and token segment lengths to pickle file
            #     with open(errs_loc + f"sys_choices_sys_dict_tok_seg_lens_seg_starts_example_{ex}.pkl", 'wb') as f:
            #         pickle.dump({
            #             'sys_choices_per_config': sys_choices_per_config,
            #             'sys_dict_per_config': sys_dict_per_config,
            #             'tok_seg_lens_per_config': tok_seg_lens_per_config,
            #             'seg_starts_per_config': seg_starts_per_config
            #         }, f)
                
            #     end = time.time()  # end the timer for needle predictions
            #     print(f"time elapsed for Needle Pred example {ex}:", (end - start) / 60, "min\n\n\n")  # print the time elapsed for needle predictions

            # for key in err_lss_examples.keys():
            #     print(f"err_lss_examples[{key}] len: {len(err_lss_examples[key])}")
            #     err_lss_examples[key] = np.array(err_lss_examples[key])
            #     print(f"err_lss_examples[{key}] shape: {err_lss_examples[key].shape}")

            # with open(errs_loc + "err_lss_examples.pkl", 'wb') as f:
            #     pickle.dump(err_lss_examples, f)

            # return None


    else:
        err_lss, irreducible_error = compute_errors(config, config.C_dist, run_deg_kf_test,
                                                wentinn_data=False, tf=tf)

    
    if run_deg_kf_test:
        # save err_lss and irreducible_error to a file
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + f"/{config.val_dataset_typ}_err_lss_deg_kf_test.pkl",
                "wb") as f:
            pickle.dump(err_lss, f)
    else:
        print("saving prediction errors")
        # save err_lss and irreducible_error to a file
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + "_"+ f"step={ckpt_steps}.ckpt" + f"/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl",
                "wb") as f:
            pickle.dump(err_lss, f)
            print("err_lss keys", err_lss.keys())

    with open(
            parent_parent_dir + "/prediction_errors" + config.C_dist + "_"+ f"step={ckpt_steps}.ckpt" + f"/{config.val_dataset_typ}_state_dim_{config.nx}_irreducible_error.pkl",
            "wb") as f:
        pickle.dump(irreducible_error, f)

    return None


def save_preds_conv_helper(save_dir, run_deg_kf_test, config):
    os.makedirs(save_dir, exist_ok=True)

    err_lss, irreducible_error = compute_errors_conv(config, config.C_dist, run_deg_kf_test,
                                                        wentinn_data=False)  # , emb_dim)

    print("helper len of irreducible_error:", len(irreducible_error))
    # save err_lss and irreducible_error to a file
    with open(
            save_dir + f"/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl",
            "wb") as f:
        pickle.dump(err_lss, f)

    with open(
            save_dir + f"/{config.val_dataset_typ}_state_dim_{config.nx}_irreducible_error.pkl",
            "wb") as f:
        pickle.dump(irreducible_error, f)
    return


def save_preds_conv(make_preds, run_deg_kf_test, config):
    # get the step size from the ckpt_path
    ckpt_steps = get_step_number(config.ckpt_path)
    print("ckpt_steps:", ckpt_steps)

    # make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    save_dir = parent_parent_dir + "/prediction_errors" + config.C_dist + "_"+ f"step={ckpt_steps}.ckpt"

    # a boolean for whether the below directory exists
    #check if a specific file named config.val_dataset_typ_state_dim_{config.nx}_err_lss.pkl exists in the directory
    if os.path.exists(save_dir + f"/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl"):
        print(f"{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl for ", ckpt_steps, " already exists")
    else:
        print(f"{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl for ", ckpt_steps, " does not exist")
        save_preds_conv_helper(save_dir, run_deg_kf_test, config)
    return

        


def load_preds(run_deg_kf_test, excess, num_systems, config):
    # make the prediction errors directory
    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)

    # get the step size from the ckpt_path
    ckpt_steps = get_step_number(config.ckpt_path)
    print("ckpt_steps:", ckpt_steps)

    if run_deg_kf_test:
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + f"/{config.val_dataset_typ}_err_lss_deg_kf_test.pkl",
                "rb") as f:
            err_lss_load = pickle.load(f)
            print("len(err_lss_load):", len(err_lss_load))
    else:
        with open(
                parent_parent_dir + "/prediction_errors" + config.C_dist + "_"+ f"step={ckpt_steps}.ckpt" + f"/{config.val_dataset_typ}_state_dim_{config.nx}_err_lss.pkl",
                "rb") as f:
            err_lss_load = pickle.load(f)

    print("err_lss_load keys:", err_lss_load.keys())

    with open(
            parent_parent_dir + "/prediction_errors" + config.C_dist + "_"+ f"step={ckpt_steps}.ckpt" + f"/{config.val_dataset_typ}_state_dim_{config.nx}_irreducible_error.pkl",
            "rb") as f:
        irreducible_error_load = pickle.load(f)

    if config.C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
        with open(f"../data/prediction_errors_unif_C/fir_bounds.pt", "rb") as f:
            fir_bounds = torch.load(f, map_location=torch.device('cpu'))
            fir_bounds = fir_bounds.T

        # with open(f"../data/prediction_errors_unif_C/wentinn_errors.pt", "rb") as f:
        #     rnn_errors = torch.load(f, map_location=torch.device('cpu'))

        # with open(f"../data/prediction_errors_unif_C/rnn_analytical_errors.pt", "rb") as f:
        #     rnn_an_errors = torch.load(f, map_location=torch.device('cpu'))
        #     rnn_an_errors = rnn_an_errors.permute(1,2,0)

        with open(f"../data/wentinn_12_04_24/errors.pt", "rb") as f:
            rnn_errors = torch.load(f, map_location=torch.device('cpu'))
            rnn_errors = rnn_errors.permute(1, 2, 0)

        with open(f"../data/wentinn_12_04_24/analytical_errors.pt", "rb") as f:
            rnn_an_errors = torch.load(f, map_location=torch.device('cpu'))
            rnn_an_errors = rnn_an_errors.permute(1, 2, 0)
    else:
        fir_bounds = np.zeros((num_systems, 1))
        rnn_errors = np.zeros((num_systems, 1, 32))
        rnn_an_errors = np.zeros((num_systems, 1, 32))

    return err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors


def setup_deg_kf_axs_arrs(num_systems):
    # create a square array of zeros that is the size of the number of systems to hold the cosine similarities
    cos_sims = np.zeros((num_systems, num_systems))
    err_ratios = np.zeros((num_systems, num_systems))
    zero_ratios = np.zeros((num_systems, num_systems))

    deg_fig = plt.figure(figsize=(40, 40))
    ax1 = deg_fig.add_subplot(331)  # This creates the first subplot
    ax2 = deg_fig.add_subplot(332)  # This creates the second subplot
    ax3 = deg_fig.add_subplot(333)  # This creates the third subplot
    ax4 = deg_fig.add_subplot(334)  # This creates the third subplot
    ax5 = deg_fig.add_subplot(335)  # This creates the third subplot
    ax6 = deg_fig.add_subplot(336)  # This creates the third subplot
    ax7 = deg_fig.add_subplot(337)  # This creates the third subplot
    ax8 = deg_fig.add_subplot(338)  # This creates the third subplot
    ax9 = deg_fig.add_subplot(339)  # This creates the third subplot
    axs = [[ax1, ax2, ax3], [ax4, ax5, ax6], [ax7, ax8, ax9]]
    return cos_sims, err_ratios, zero_ratios, deg_fig, axs


def create_plots(config, model, run_preds, run_deg_kf_test, excess, num_systems, shade, logscale, train_conv, tf, ys, sim_objs, run_kf_ols=True):
    C_dist = config.C_dist
    
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)

    if run_preds:
        # print("config path:", config.ckpt_path)
        save_preds(run_deg_kf_test, config, model, train_conv, tf, ys, sim_objs, run_kf_ols=run_kf_ols)  # save the predictions to a file

        if train_conv:
            return None

    # load the prediction errors from the file
    if config.multi_sys_trace:
        return None 
        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures/multi_sys_trace/", exist_ok=True)

        # get the step size from the ckpt_path
        ckpt_steps = get_step_number(config.ckpt_path)
        print("ckpt_steps:", ckpt_steps)

        print(f"\n\n config.val_dataset_typ: {config.val_dataset_typ} in create_plots")

        errs_dir = parent_parent_dir + f"/prediction_errors{config.C_dist}_step={ckpt_steps}.ckpt"
        errs_loc = errs_dir + f"/" + ("single_system_" if config.single_system else "") + ("zero_cut_" if config.zero_cut else "") + (f"needle_haystack_len_{config.num_sys_haystack}_{config.datasource}_" if config.needle_in_haystack else "") + ("fin_seg_ext_" if config.needle_in_haystack and config.needle_final_seg_extended else "") + f"{config.val_dataset_typ}_state_dim_{config.nx}_"

        #load the system indices, starting indices, and token segment lengths from the pickle file
        with open(errs_loc + "sys_choices_sys_dict_tok_seg_lens_seg_starts" + ("_example_0" if config.needle_in_haystack else "") + ".pkl", 'rb') as f:
            data = pickle.load(f)
            sys_choices_per_config = data['sys_choices_per_config']
            sys_dict_per_config = data['sys_dict_per_config']
            tok_seg_lens_per_config = data['tok_seg_lens_per_config']
            seg_starts_per_config = data['seg_starts_per_config']

        #load the err_lss dict from the pkl file
        with open(
                errs_loc + "err_lss.pkl",
                "rb") as f:
            err_lss_load = pickle.load(f)

        for trace_conf in range(len(seg_starts_per_config)):
            fig = plt.figure(figsize=(40, 15)) # create a figure with a size of 15x15
            ax = fig.add_subplot(111)

            handles = plot_errs_multi_sys(trace_conf, err_lss_load, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, ax=ax)

            ax.legend(fontsize=16, loc="upper right", ncol=max(1, math.floor(len(handles) / 2)))
            ax.set_xlabel("Context", fontsize=30)

            ax.set_ylabel("MSE", fontsize=30)
            # ax.set_ylabel("Err - Empirical KF Err" if logscale else "Prediction Error", fontsize=30)
            # ax.set_ylabel("Median Error" if logscale else "Avg Error", fontsize=30)

            ax.grid(which="both")
            if logscale:
                ax.set_yscale('log')
                ax.set_xscale('log')

            ax.set_title(("NoPE " if not config.use_pos_emb else "") + ("Gaussian Systems " if config.val_dataset_typ == "gaussA" else ("Orthogonal Systems " if config.val_dataset_typ == "ortho" else ("Identity Systems " if config.val_dataset_typ == "ident" else "")))   + f"MSE vs Context. Trace Configuration: {trace_conf}")
            # ax.set_ylim([0,2])
            # Set major and minor gridlines

            ax.set_xlim(left=0, right=251)
            # ax.set_ylim(bottom=0, top=5.5)  # set the y axis limits


            # Optionally, customize major and minor ticks
            ax.minorticks_on()

            # Set minor vertical grid lines to be on intervals of 1
            # Set major ticks on every interval of 50
            ax.set_xticks(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 50))

            # Set minor vertical grid lines to be on intervals of 1
            ax.set_xticks(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 1), minor=True)

            ax.tick_params(axis='both', which='major', length=7, width=1, labelsize=30)
            ax.tick_params(axis='both', which='minor', length=4, width=0.5, labelsize=0)
            ax.grid(which='major', linestyle='-', linewidth=1)
            ax.grid(which='minor', linestyle='--', linewidth=0.5)

            #add the date and time to the filename
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")

            #get only the ckpt step from the ckpt_path
            ckpt_step = config.ckpt_path.split("=")[1].split(".")[0]

            #add a caption to the bottom of the figure
            fig.text(0.5, 0.01, "step=" + ckpt_step + "_" + timestamp, ha='center', fontsize=30)
            os.makedirs(parent_parent_dir + f"/figures/multi_sys_trace/"+ (f"needle_in_haystack_example_0/{config.datasource}/" if config.needle_in_haystack else ""), exist_ok=True)
            fig.savefig(
                parent_parent_dir + f"/figures/multi_sys_trace/" + (f"needle_in_haystack_example_0/{config.datasource}/" if config.needle_in_haystack else "") + ("fin_seg_ext_" if config.needle_in_haystack and config.needle_final_seg_extended else "") + ("NoPE_" if not config.use_pos_emb else "") + ("single_system_" if config.single_system else "") + f"{config.val_dataset_typ}{C_dist}_trace_conf_{trace_conf}" + (
                    "_logscale" if logscale else "") + f"_step={ckpt_step}_" + timestamp) 
        return None


    else:
        err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess,
                                                                                             num_systems, config)


   # Define the dark colors in hex format
    colors = [
    '#1f77b4',  # Dark Blue
    '#ff7f0e',  # Dark Orange
    '#2ca02c',  # Dark Green
    '#d62728',  # Dark Red
    '#9467bd',  # Purple
    '#17becf',  # Dark Cyan
    '#8c564b',  # Dark Brown
]

    print("len(err_lss_load):", len(err_lss_load))
    for sys in range(len(irreducible_error_load)):

        fig = plt.figure(figsize=(15, 15)) # create a figure with a size of 15x15
        ax = fig.add_subplot(111)
        # plot transformer, KF and FIR errors
        handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade,
                                        normalized=logscale)
        
        
        ax.legend(fontsize=16, loc="upper right", ncol=max(1, math.floor(len(handles) / 2)))
        ax.set_xlabel("i", fontsize=30)

        ax.set_ylabel("Median of Err / Empirical KF Err" if logscale else "Prediction Error", fontsize=30)
        # ax.set_ylabel("Err - Empirical KF Err" if logscale else "Prediction Error", fontsize=30)
        # ax.set_ylabel("Median Error" if logscale else "Avg Error", fontsize=30)

        ax.grid(which="both")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        if logscale:
            ax.set_yscale('log')
            ax.set_xscale('log')

        # ax.set_yscale('linear')
        # ax.set_xscale('linear')


        # if not logscale:
        #     ax.set_ylim(bottom=10 ** (-0.7), top=0.5 * 10 ** (1))  # set the y axis limits

        ax.set_title("System " + str(sys) + (": Rotated Diagonal (|N(0,1)| <= 0.95 Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Unif(-1,1) Eigs) A " if config.val_dataset_typ == "rotDiagA_unif" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
            ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                            "Uniform C" if C_dist == "_unif_C" else (
                                "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
        # ax.set_xlim(left=0, right=10)

        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        #add the date and time to the filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        #get only the ckpt step from the ckpt_path
        ckpt_step = config.ckpt_path.split("=")[1].split(".")[0]

        #add a caption to the bottom of the figure
        fig.text(0.5, 0.01, "step=" + ckpt_step + "_" + timestamp, ha='center', fontsize=30)
        fig.savefig(
            parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                "logscale" if logscale else "") + "_" + "step=" + ckpt_step + "_" + timestamp) 

    return None

# Sort handles and labels based on "MOP" part or "_analytical" part
# Extracting the number after "MOP" or "_analytical" and using it for sorting
def extract_sort_key(label):
    if "_analytical" in label:
        return int(label.split("_analytical")[1])
    else:
        return int(label.split("MOP")[1])


def convergence_plots(j, config, run_preds, run_deg_kf_test, kfnorm, num_systems, shade, fig, ax, ts, kal_errors):
    excess = False
    C_dist = config.C_dist
    print("\n\n", "config path:", config.ckpt_path)
    if run_preds:
        print("\n\nRunning predictions")
        save_preds_conv(run_preds, run_deg_kf_test, config)  # save the predictions to a file
    print("\n\nLoading predictions")
    # load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess,
                                                                                             num_systems, config)

    colors = [
        '#D32F2F',  # Red
        '#C2185B',  # Pink
        '#7B1FA2',  # Purple
        '#512DA8',  # Deep Purple
        '#303F9F',  # Indigo
        '#1976D2',  # Blue
        '#0288D1',  # Light Blue
        '#0097A7',  # Cyan
        '#00796B',  # Teal
        '#388E3C',  # Green
        '#689F38',  # Light Green
        '#AFB42B',  # Lime
        '#FBC02D',  # Yellow
        '#FFA000',  # Amber
        '#F57C00',  # Orange
        '#E64A19',  # Deep Orange
        '#5D4037',  # Brown
        '#616161',  # Grey
        '#455A64',  # Blue Grey
        '#8E24AA',  # Purple 600
        '#D81B60',  # Pink 600
        '#3949AB',  # Indigo 600
        '#F4511E',  # Deep Orange 600
        '#6D4C41',  # Brown 600
        '#1B5E20',  # Dark Green
        '#33691E',  # Lime Green Dark
        '#827717',  # Olive
        '#F9A825',  # Mustard
        '#FF6F00',  # Orange Deep
        '#E65100',  # Orange Dark
        '#BF360C',  # Deep Orange Dark
        '#3E2723',  # Deep Brown
        '#212121',  # Almost Black
        '#263238',  # Blue Grey Dark
        '#004D40',  # Teal Dark
        '#006064',  # Cyan Dark
        '#01579B',  # Light Blue Dark
        '#0D47A1',  # Blue Dark
        '#1A237E',  # Indigo Dark
        '#311B92',  # Deep Purple Dark
        '#4A148C',  # Purple Dark
        '#880E4F',  # Pink Dark
        '#B71C1C',  # Red Dark
        '#D50000',  # Red Accent
        '#C51162',  # Pink Accent
        '#AA00FF',  # Purple Accent
        '#6200EA',  # Deep Purple Accent
        '#304FFE',  # Indigo Accent
    ]
    print("\n\nPlotting predictions")
    sys_errs = []
    sys_errs_an = []
    for sys in range(len(irreducible_error_load)):
        # plot transformer, KF and FIR errors
        # get the checkpoint steps number from the checkpoint path
        ckpt_steps = config.ckpt_path.split("step=")[1].split(".")[0]  # get the checkpoint steps number
        handles, err_avg_t, err_avg_t_an = plot_errs_conv(ts, j, colors, sys, err_lss_load, irreducible_error_load, ckpt_steps,
                                            kfnorm, ax=ax[sys], shade=shade, kal_err=kal_errors)  # plot the errors
        sys_errs.append(err_avg_t)  # append the system number and the error average at step t
        sys_errs_an.append(err_avg_t_an)  # append the system number and the error average at step t

        # Step 1: Collect legend handles and labels
        handles, labels = ax[sys].get_legend_handles_labels()  # handles and labels of the legend

        # Step 2: Sort handles and labels
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: extract_sort_key(hl[1]))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)

        # Step 3: Create the legend with sorted handles and labels
        ax[sys].legend(sorted_handles, sorted_labels, fontsize=18, loc="upper right", ncol=1)

        # ax[sys].legend(fontsize=18, loc="upper right", ncol=1)
        ax[sys].set_xlabel("t", fontsize=30)
        ax[sys].set_ylabel("Prediction Error", fontsize=30)
        ax[sys].grid(which="both")
        ax[sys].tick_params(axis='both', which='major', labelsize=30)
        ax[sys].tick_params(axis='both', which='minor', labelsize=20)

        # set y axis limits
        ax[sys].set_ylim(bottom=10 ** (-2), top=5 * 10 ** (1))

        ax[sys].set_title("System " + str(sys) + (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
            ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                              "Uniform C" if C_dist == "_unif_C" else (
                                  "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")) + (
                              " Normalized" if kfnorm else ""), fontsize=20)
        # ax.set_xlim(left=0, right=10)

    # get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    # get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
    fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_conv" + (
        "_normalized" if kfnorm else "") + ("-changing" if config.changing else ""))

    return (ckpt_steps, sys_errs), (ckpt_steps, sys_errs_an)  # return the checkpoint steps number and the system errors


####################################################################################################
# main function
if __name__ == '__main__':
    config = Config()

    C_dist = "_gauss_C"  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var"
    run_preds = True  # run the predictions evaluation
    run_deg_kf_test = False  # run degenerate KF test
    excess = False  # run the excess plots
    if excess:
        fig = plt.figure(figsize=(30, 15))
        ax = fig.add_subplot(111)
    shade = False

    num_systems = config.num_val_tasks  # number of validation tasks

    if run_preds:
        print("config path:", config.ckpt_path)
        save_preds(run_deg_kf_test, config)  # save the predictions to a file

    # load the prediction errors from the file
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess,
                                                                                             num_systems, config)

    if run_deg_kf_test:
        cos_sims, err_ratios, zero_ratios, deg_fig, axs = setup_deg_kf_axs_arrs(num_systems)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#A80000', '#bcbd22']

    print("len(err_lss_load):", len(err_lss_load))
    for sys in range(len(irreducible_error_load)):

        if run_deg_kf_test:
            for i in range(num_systems):
                err_lss_copy = copy.deepcopy(err_lss_load)
                err_lss_copy["Kalman"] = err_lss_copy["Kalman"][i]

                print("KF trained on system", i, "testing on system", sys)

                # plot transformer, KF and FIR errors
                handles, err_rat = plot_errs(colors, sys, err_lss_copy, irreducible_error_load, ax=axs[i][sys],
                                             shade=True, normalized=excess)
                print("err_rat:", err_rat)

                err_ratios[i, sys] = err_rat[0]
                zero_ratios[i, sys] = err_rat[1]

                # compute the cosine similarity between the err_lss_load["Kalman"][i][sys] and err_lss_load[i][i]
                cos_sim = np.dot(err_lss_load["Kalman"][i][sys].flatten(), err_lss_load["Kalman"][i][i].flatten()) / (
                            np.linalg.norm(err_lss_load["Kalman"][i][sys]) * np.linalg.norm(
                        err_lss_load["Kalman"][sys][sys]))
                print("cosine similarity between KF trained on system", i, "testing on system", sys,
                      "and KF trained and tested on system", sys, ":", cos_sim)
                cos_sims[i, sys] = cos_sim

                if C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
                    # plot fir bounds
                    for j in range(fir_bounds.shape[1] - 2):
                        handles.extend(axs[i][sys].plot(np.array(range(config.n_positions)),
                                                        fir_bounds[sys, j] * np.ones(config.n_positions),
                                                        label="IR Analytical Length " + str(j + 1), linewidth=3,
                                                        linestyle='--', color=colors[j + 5]))

                    # plot RNN errors
                    avg, std = rnn_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_errors.shape[1])) * rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(
                        axs[i][sys].scatter(np.arange(0, 32 * 5, 5), avg_numpy, label="RNN", linewidth=1, marker='x',
                                            s=50, color=colors[len(err_lss_copy)]))
                    axs[i][sys].fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy,
                                             avg_numpy + std_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_an_errors.shape[1])) * rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(
                        axs[i][sys].scatter(np.arange(0, 251, 5), avg_an_numpy, label="RNN Analytical", linewidth=1,
                                            marker='o', s=100, color=colors[len(err_lss_copy)], zorder=10))
                    axs[i][sys].fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy,
                                             avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0],
                                             alpha=0.2)

                del err_lss_copy  # Delete the variable
                gc.collect()  # Start the garbage collector

                axs[i][sys].legend(fontsize=18, loc="upper right", ncol=math.floor(len(handles) / 4))
                axs[i][sys].set_xlabel("t", fontsize=30)
                axs[i][sys].set_ylabel("Prediction Error", fontsize=30)
                axs[i][sys].grid(which="both")
                axs[i][sys].tick_params(axis='both', which='major', labelsize=30)
                axs[i][sys].tick_params(axis='both', which='minor', labelsize=20)
                axs[i][sys].set_title("KF system " + str(i) + " testing on system " + str(sys) + (
                    ": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                        ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else ": Dense A ")) + (
                                          "Uniform C" if C_dist == "_unif_C" else (
                                              "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))
                axs[i][sys].set_ylim(bottom=10 ** (-0.7), top=2 * 10 ** (0))
                # axs[i][sys].set_xlim(left=0, right=10)

            if sys == num_systems - 1 and i == num_systems - 1:
                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                deg_fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + (
                    "-changing_deg_kf_test" if config.changing else "deg_kf_test"))
        else:
            fig = plt.figure(figsize=(15, 9))
            ax = fig.add_subplot(111)
            # plot transformer, KF and FIR errors
            handles, err_rat = plot_errs(colors, sys, err_lss_load, irreducible_error_load, ax=ax, shade=shade,
                                         normalized=excess)

            if C_dist == "_unif_C" and config.val_dataset_typ == "ypred":
                if excess:
                    # plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)),
                                               (fir_bounds[sys, i] - irreducible_error_load[sys]) * np.ones(
                                                   config.n_positions),
                                               label="IR Analytical Length " + str(i + 1) + " sys: " + str(sys),
                                               linewidth=3, linestyle='--'))  # , color = colors[i + 5]))

                    # #plot RNN errors
                    # rnn_er = rnn_errors[sys].detach().numpy()
                    # kalman_err = err_lss_load["Kalman"][sys,:, ::5].mean(axis=(0))
                    # #figure out how to take median and quantiles of the rnn errors
                    # rnn_q1, rnn_median, rnn_q3 = np.quantile((rnn_er -kalman_err), [0.25, 0.5, 0.75], axis=-2)
                    # scale = rnn_median[1]
                    # rnn_median = rnn_median/scale
                    # rnn_q1 = rnn_q1/scale
                    # rnn_q3 = rnn_q3/scale
                    # N = rnn_median.shape[0]
                    # # Adjust the range of np.arange function
                    # x = np.arange(1, (N-1)*5 + 1, 5)
                    # handles.append(ax.scatter(x, rnn_median[1:], label="RNN sys: " + str(sys), linewidth=3, marker='x', s=50))#, color=colors[len(err_lss_load)]))
                    # if shade:
                    #     ax.fill_between(x, rnn_q1[1:], rnn_q3[1:], facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    print("rnn_an_errors.shape:", rnn_an_errors.shape)
                    # plot Analytical RNN errors
                    rnn_an_er = rnn_an_errors[sys].detach().numpy()
                    kalman_err = err_lss_load["Kalman"][sys, :, ::5].mean(axis=(0))
                    # figure out how to take median and quantiles of the rnn errors
                    rnn_an_q1, rnn_an_median, rnn_an_q3 = np.quantile((rnn_an_er - kalman_err), [0.25, 0.5, 0.75],
                                                                      axis=-2)
                    scale = rnn_an_median[1]
                    rnn_an_median = rnn_an_median / scale
                    rnn_an_q1 = rnn_an_q1 / scale
                    rnn_an_q3 = rnn_an_q3 / scale
                    N = rnn_an_median.shape[0]
                    # Adjust the range of np.arange function
                    x = np.arange(1, (N - 1) * 5 + 1, 5)
                    handles.append(
                        ax.scatter(x, rnn_an_median[1:], label="RNN Analytical sys: " + str(sys), linewidth=1,
                                   marker='o', s=100))  # , color=colors[len(err_lss_load)]))
                    if shade:
                        ax.fill_between(x, rnn_an_q1[1:], rnn_an_q3[1:], facecolor=handles[-1].get_facecolor()[0],
                                        alpha=0.2)
                    # avg_an, std_an = rnn_an_errors[sys,:,:].mean(axis=(0)), (3/np.sqrt(rnn_an_errors.shape[1]))*rnn_an_errors.std(axis=(0, 1))
                    # avg_an_numpy = avg_an.detach().numpy()
                    # std_an_numpy = std_an.detach().numpy()
                    # handles.append(ax.scatter(np.arange(0,251,5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o', s=100))#, color=colors[len(err_lss_load)], zorder=10))
                    # ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy, avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)
                else:
                    # plot fir bounds
                    for i in range(fir_bounds.shape[1] - 2):
                        handles.extend(ax.plot(np.array(range(config.n_positions)),
                                               fir_bounds[sys, i] * np.ones(config.n_positions),
                                               label="IR Analytical Length " + str(i + 1), linewidth=3, linestyle='--',
                                               color=colors[i + 5]))

                    # plot RNN errors
                    print("rnn_errors.shape:", rnn_errors.shape)
                    avg, std = rnn_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_errors.shape[1])) * rnn_errors.std(axis=(0, 1))
                    avg_numpy = avg.detach().numpy()
                    std_numpy = std.detach().numpy()
                    handles.append(
                        ax.scatter(np.arange(0, config.n_positions + 1, 5), avg_numpy, label="RNN", linewidth=1,
                                   marker='x', s=50, color=colors[len(err_lss_load)]))
                    ax.fill_between(np.arange(rnn_errors.shape[-1]), avg_numpy - std_numpy, avg_numpy + std_numpy,
                                    facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

                    avg_an, std_an = rnn_an_errors[sys, :, :].mean(axis=(0)), (3 / np.sqrt(
                        rnn_an_errors.shape[1])) * rnn_an_errors.std(axis=(0, 1))
                    avg_an_numpy = avg_an.detach().numpy()
                    std_an_numpy = std_an.detach().numpy()
                    handles.append(
                        ax.scatter(np.arange(0, 251, 5), avg_an_numpy, label="RNN Analytical", linewidth=1, marker='o',
                                   s=100, color=colors[len(err_lss_load)], zorder=10))
                    ax.fill_between(np.arange(rnn_an_errors.shape[-1]), avg_an_numpy - std_an_numpy,
                                    avg_an_numpy + std_an_numpy, facecolor=handles[-1].get_facecolor()[0], alpha=0.2)

            if excess:
                ncol = 1 if len(handles) < 4 else math.floor(len(handles) / 4)
                ax.legend(fontsize=18, loc="lower left", ncol=ncol)
                ax.set_xlabel("log(t)", fontsize=30)
                ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                # make the x axis log scale
                ax.set_xscale('log')
                # ax.set_ylim(bottom=-1, top=2*10**(-1))
                ax.set_title("System " + str(sys) + (": Rotated Diagonal (|N(0,1)| <= 0.95 Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Unif(-1,1) Eigs) A " if config.val_dataset_typ == "rotDiagA" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
                # ax.set_xlim(left=0, right=10)

                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(
                    parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                        "-changing" if config.changing else "_excess"))
            else:
                ax.legend(fontsize=18, loc="upper right", ncol=max(1, math.floor(len(handles) / 4)))
                ax.set_xlabel("t", fontsize=30)
                ax.set_ylabel("Prediction Error", fontsize=30)
                ax.grid(which="both")
                ax.tick_params(axis='both', which='major', labelsize=30)
                ax.tick_params(axis='both', which='minor', labelsize=20)
                ax.set_ylim(bottom=10 ** (-0.7), top=3 * 10 ** (0))
                ax.set_title("System " + str(sys) + (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")), fontsize=20)
                # ax.set_xlim(left=0, right=10)

                # get the parent directory of the ckpt_path
                parent_dir = os.path.dirname(config.ckpt_path)

                # get the parent directory of the parent directory
                parent_parent_dir = os.path.dirname(parent_dir)
                os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
                fig.savefig(
                    parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff_" + str(sys) + (
                        "-changing" if config.changing else ""))

    if run_deg_kf_test:
        # Create a DataFrame from the numpy array
        # create a list of strings that correspond to the system numbers
        test_col = ["Test sys " + str(i) for i in range(num_systems)]
        train_col = ["Train sys " + str(i) for i in range(num_systems)]

        df = pd.DataFrame(cos_sims, columns=test_col, index=train_col)

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Cosine Similarities of KF Predictions')

        # Create a table and save it as an image
        tbl = table(ax, df, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        # get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)

        # get the parent directory of the parent directory
        parent_parent_dir = os.path.dirname(parent_dir)
        os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_cosine_similarities_deg_kf_test")

        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of KF Predictions')
        # Create a table and save it as an image
        df2 = pd.DataFrame(err_ratios, columns=test_col, index=train_col)
        tbl = table(ax, df2, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_error_ratios_deg_kf_test")

        print("zero_ratios:", zero_ratios)
        fig, ax = plt.subplots(figsize=(12, 2))  # set size frame
        ax.axis('off')
        ax.set_title('Error Ratios of Zero Predictions')
        # Create a table and save it as an image
        df3 = pd.DataFrame(zero_ratios[0, :].reshape(1, -1), columns=test_col)
        tbl = table(ax, df3, loc='center', cellLoc='center')
        tbl.scale(1, 1.5)

        plt.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_zero_ratios_deg_kf_test")

    if excess:
        ncol = 1 if len(handles) < 4 else math.floor(len(handles) / 2)
        ax.legend(fontsize=14, loc="lower left", ncol=ncol)
        ax.set_xlabel("log(t)", fontsize=30)
        ax.set_ylabel("log(Prediction Error - Emp Kalman Error)", fontsize=20)
        ax.grid(which="both")
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        # make the x axis log scale
        ax.set_xscale('log')
        # ax.set_ylim(bottom=-1, top=2*10**(-1))
        ax.set_title("System " + str(sys) + (": Rotated Diagonal (|N(0,1)| <= 0.95 Eigs) A " if config.val_dataset_typ == "rotDiagA_gauss" else (": Rotated Diagonal (Unif(-1,1) Eigs) A " if config.val_dataset_typ == "rotDiagA" else (": Rotated Diagonal A " if config.val_dataset_typ == "rotDiagA" else (
                    ": Upper Triangular A " if config.val_dataset_typ == "upperTriA" else (
                        ": N(0,0.33) A " if config.val_dataset_typ == "gaussA" else ": Dense A "))) + (
                                 "Uniform C" if C_dist == "_unif_C" else (
                                     "N(0,0.33) C" if C_dist == "_gauss_C" else "N(0,1) C")))))
        # ax.set_xlim(left=0, right=10)
        os.makedirs(parent_parent_dir + f"/figures", exist_ok=True)
        fig.savefig(parent_parent_dir + f"/figures/{config.val_dataset_typ}" + C_dist + "_system_cutoff" + (
            "-changing" if config.changing else "_excess"))
