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
# import sys
# sys.path.append(os.path.abspath('../../src'))

from data_processing import gen_ckpt_steps, move_dict_to_device, get_other_err, get_mop_ratios_ckpt, compute_ratio
# sys.path.append(os.path.abspath('..'))

from check_ecdf import get_empirical_cdf

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def get_seg_starts_per_config(experiment, valA, valC, state_dim, ckpt, print_seg_starts=False):
    # load the sys choices etc
    errs_dir = "../outputs/GPT2/" + experiment + f"/prediction_errors{valC}_step={ckpt}.ckpt"
    errs_loc = errs_dir + f"/single_system_{valA}_state_dim_{state_dim}_sys_choices_sys_dict_tok_seg_lens_seg_starts.pkl"

    if not os.path.exists(errs_loc):
        return None
    else:
        with open(errs_loc, "rb") as f:
            data = pickle.load(f)
            seg_starts_per_config = data['seg_starts_per_config']
            if print_seg_starts:
                print(f"seg_starts_per_config: {seg_starts_per_config}")
                
        return seg_starts_per_config

def train_conv_plots(experiments, trainAs, kal_ckpt, valA, C_dist, num_val_systems, compute_more_ckpts=False, ind=250, min_ckpt=79, max_ckpt=79000, interval=79, nx=10, needle_in_haystack=False, single_system=False):
    num_preds = 3 #len(experiments) #number of predictors to plot
    colors = plt.cm.tab10(np.linspace(0, 1, num_preds))

    plot_time = time.ctime()

    #create a figure with subplots for each of the m indexes for the cdfs
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True)
    filename = f'training_dist_comparison_val_{valA}_state_dim_{nx}_val_sys_{num_val_systems}_{time.time()}.png'

    parent_path = "../outputs/GPT2/"

    filepath = os.path.abspath(f"../outputs/train_conv/{filename}")
    print(filepath)

    ckpt_steps = gen_ckpt_steps(min_ckpt, max_ckpt, interval)

    i = 0
    for experiment in experiments:
        if not os.path.exists(parent_path + experiment + "/train_conv/quantiles.npz") or compute_more_ckpts:
            pred_ckpts = []
            quantiles = []
            print("\n\ni", i)
            if not needle_in_haystack:
                kal_err = get_other_err(valA, C_dist, kal_ckpt[i], experiment, "Kalman", nx=nx, single_system=single_system)
            for ckpt_step in ckpt_steps:
                mop_err, pred_ckpt = get_mop_ratios_ckpt(valA, C_dist, ckpt_step, experiment, nx=nx, single_system=single_system)
                if pred_ckpt:
                
                    if needle_in_haystack:
                        kal_err = get_other_err(valA, C_dist, ckpt_step, experiment, "Kalman", nx=nx, single_system=single_system)
                    quantile = compute_ratio(ind=ind, err=mop_err, kalman_err=kal_err)
                    if single_system:

                        print(f"quantile shape before seg start choice: {quantile.shape}")
                        seg_starts_per_config = get_seg_starts_per_config(experiment, valA, C_dist, nx, ckpt_step, print_seg_starts=True)
                        seg_starts = seg_starts_per_config[0]

                        if len(seg_starts) > 1:
                            quantile = quantile[:, seg_starts[1] + 1] #take the quantile at the start of the second segment
                            print(f"seg_starts[1] + 1: {seg_starts[1] + 1}")
                            print(f"quantile shape after seg start choice: {quantile.shape}")
                        else:
                            print("only one segment start so disregard ckpt")
                            continue
                        

                    
                    pred_ckpts.append(pred_ckpt)
                    print(f"quantile shape: {quantile.shape}")
                    if isinstance(quantile, torch.Tensor):
                        quantile = quantile.cpu().numpy()
                    del mop_err
                    quantiles.append(quantile)

                    torch.cuda.empty_cache()
                    gc.collect()
            del kal_err
            torch.cuda.empty_cache()
            gc.collect()

            quantiles = np.array(quantiles)
            
            #save quantiles to file
            os.makedirs(parent_path + experiment + "/train_conv", exist_ok=True)
            np.savez_compressed(parent_path + experiment + "/train_conv/quantiles.npz", pred_ckpts=pred_ckpts, quantiles=quantiles)
        else:
            data = np.load(parent_path + experiment + "/train_conv/quantiles.npz", allow_pickle=False)
            pred_ckpts = data["pred_ckpts"]
            quantiles = data["quantiles"]

        # quantiles -= 1
        print("quantiles shape", quantiles.shape)    
        ##plotting stuff
        ax.plot(pred_ckpts, quantiles[:,1], marker="*", linewidth=3, color= colors[i], label=trainAs[i] + " Median")# label= f"Experiment: {experiments[i]} Median")
        plt.fill_between(pred_ckpts, quantiles[:,0], quantiles[:,2], color=colors[i], alpha=0.2) #, label='25th-75th Percentile Range')
        torch.cuda.empty_cache()
        gc.collect()

        ax.set_title(f"Error Ratio of Median Test System vs Training Iteration: Gaussian Test Distribution.")
        ax.grid(True)
        ax.set_ylabel("Error of Median Test System / Emp Kal Error")
        ax.set_xlabel("Training Iteration")
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        ax.legend()
        # ax.set_yscale("log")
        # ax.set_xscale("log")

        fig.text(0.5, 0.01, f'Generated at {plot_time}', ha='center')

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        #save the figures
        fig.savefig(filepath)
        if i ==0:
            lr_med = quantiles[1,:]
            lr_pred_ckpts = pred_ckpts

        i+=1

    return lr_med, lr_pred_ckpts