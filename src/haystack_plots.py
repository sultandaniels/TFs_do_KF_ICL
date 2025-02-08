import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

import logging
import pickle
from datetime import datetime
import os
from data_processing import gen_ckpt_steps
from conv_plots_funcs import get_seg_starts_per_config
from get_last_checkpoint import split_path
from get_last_checkpoint import get_last_checkpoint
import torch
import gc



def comp_quartiles(err_lss_examples, ratio=False, train_conv=False, kal_err=None):
    quartiles = {}
    if ratio:
        if not train_conv:
            kal_err = err_lss_examples["Kalman_rem"]

    for key in err_lss_examples.keys():
        if not (key == "Analytical_Kalman" or key == "Kalman_rem" or key == "Kalman"):
            if ratio:
                rat = err_lss_examples[key] / kal_err
            else:
                rat = err_lss_examples[key]
            med = np.median(rat, axis=2)
            # print(f"shape of med: {med.shape}")
            quartiles[key] = np.percentile(med, [25,50,75], axis=0)
            # print(f"shape of quartiles[{key}]: {quartiles[key].shape}")
            
    return quartiles

def save_quartiles(quartiles_file, quartiles, seg_ext_quartiles_file, seg_ext_quartiles):
    os.makedirs(os.path.dirname(quartiles_file), exist_ok=True)
    np.savez(quartiles_file, **quartiles)

    os.makedirs(os.path.dirname(seg_ext_quartiles_file), exist_ok=True)
    np.savez(seg_ext_quartiles_file, **seg_ext_quartiles)
    return None

def load_quartiles(model_dir, experiment, valC, ckpt_step, valA, state_dim, datasource):
    quartiles = None
    seg_ext_quartiles = None

    quartiles_file = model_dir + experiment + "/needles/quartiles.npz"

    seg_ext_quartiles_file = model_dir + experiment + "/needles/seg_ext_quartiles.npz"

    if os.path.exists(quartiles_file):
        print(f"Loading quartiles from {quartiles_file}")
        quartiles = np.load(quartiles_file)

    if os.path.exists(seg_ext_quartiles_file):
        print(f"Loading seg ext quartiles from {seg_ext_quartiles_file}")
        seg_ext_quartiles = np.load(seg_ext_quartiles_file)


    return quartiles_file, seg_ext_quartiles_file, quartiles, seg_ext_quartiles
 
def plot_needle_position(experiment, datasource, state_dim, ckpt_step, valA, valC, haystack_len, steps_in, open_paren_ind, quartiles, seg_ext_quartiles, colors, nope):

    real_steps = [x + open_paren_ind for x in steps_in]
    real_steps_ext = [x + open_paren_ind-2 for x in steps_in]

    
    if valA == "gaussA":
        num_axes = len(steps_in)
    else:
        num_axes = 1
        
    fig, ax = plt.subplots(num_axes, 1, sharex=True, figsize=(5, 2.7*len(steps_in))) #

    print(f"real_steps: {real_steps}, real_steps_ext: {real_steps_ext}")


    for needle in range(haystack_len):
        key_count = 0
        for key in quartiles.keys():
            print(f"key_count: {key_count}")
            if "OLS" not in key and "Simulation" not in key:
                # ax[needle].scatter(quartiles[key][1, needle], label=key)
                step_count = 0
                for step in steps_in:

                    key_label = "TF" if key == "MOP" else key

                    print(f"needle: {needle}, step: {step}, real_step: {real_steps[step_count]}")

                    y = quartiles[key][1, needle, real_steps[step_count]]
                    print(f"y: {y}")
                    if valA == "gaussA":
                        y -= 1
                    yerr = [
                        quartiles[key][1, needle, real_steps[step_count]] - quartiles[key][0, needle, real_steps[step_count]],
                        quartiles[key][2, needle, real_steps[step_count]] - quartiles[key][1, needle, real_steps[step_count]]
                    ]#

                    yerr = np.array([[yerr[0]], [yerr[1]]])
                    if valA == "gaussA":
                        ax[step_count].errorbar(
                            haystack_len - needle - 1,
                            y,
                            yerr=yerr,  # Convert yerr to a (2, n) array-like structure
                            fmt='o',
                            label=f"{key_label}" if needle == 0 else "_nolegend_",
                            capsize=5,
                            zorder=haystack_len if key == "MOP" else 0, color=colors[key_count],
                            linewidth=2
                            )
                    else:
                        if key == "Zero":
                            color = colors[1]
                        elif step == 1:
                            color = colors[0]
                        elif step == 2:
                            color = colors[2]
                        elif step == 10:
                            color = colors[3]
                        elif step == 3:
                            color = colors[4]
                        elif step == 5:
                            color = colors[5]
                            
                        ax.errorbar(
                        haystack_len - needle - 1,
                        y,
                        yerr=yerr,  # Convert yerr to a (2, n) array-like structure
                        fmt='o',
                        label=((f"{key_label}" + (f": {step} After Open" if key == "MOP" else "")) if (needle == 0 and key == "MOP") or (needle == 0 and step == 1) else "_nolegend_"),
                        capsize=5,
                        zorder=haystack_len if key == "MOP" else 0, color=color,
                        linewidth=2
                        )

                    step_count += 1
                key_count += 1

    key_count = 0
    for key in seg_ext_quartiles.keys():
        if "OLS" not in key and "Simulation" not in key:
            step_count = 0
            for step in steps_in:
                y = seg_ext_quartiles[key][1, 0, real_steps_ext[step_count]]
                if valA == "gaussA":
                    y -= 1
                yerr = [
                    seg_ext_quartiles[key][1, 0, real_steps_ext[step_count]] - seg_ext_quartiles[key][0, 0, real_steps_ext[step_count]],
                    seg_ext_quartiles[key][2, 0, real_steps_ext[step_count]] - seg_ext_quartiles[key][1, 0, real_steps_ext[step_count]]
                ]

                yerr = np.array([[yerr[0]], [yerr[1]]])
                
                if valA == "gaussA":
                    ax[step_count].errorbar(
                        -2,
                        y,
                        yerr=yerr,  # Convert yerr to a (2, n) array-like structure
                        fmt='o',
                        label="_nolegend_",
                        capsize=5,
                        zorder=haystack_len if key == "MOP" else 0, color = colors[key_count], 
                        linewidth=2
                    )
                else:
                    if key == "Zero":
                        color = colors[1]
                    elif step == 1:
                        color = colors[0]
                    elif step == 2:
                        color = colors[2]
                    elif step == 10:
                        color = colors[3]
                    ax.errorbar(
                        -2 + step_count*0.02,
                        y,
                        yerr=yerr,  # Convert yerr to a (2, n) array-like structure
                        fmt='o',
                        label="_nolegend_",
                        capsize=5,
                        zorder=haystack_len if key == "MOP" else 0, color = color, 
                        linewidth=2
                    )

                if valA != "gaussA":
                    ax.legend(fontsize = 8, ncol=1, columnspacing=0.25)#, loc="upper left")
                    ax.set_xlabel("Needle Position from the End of the Haystack", fontsize=12, fontname="Times New Roman")
                    ax.set_ylabel(("(" if valA== "gaussA" else "") + "Error" + (" Ratio" if valA == "gaussA" else ""), fontsize=12)
                    ax.set_xlim(-3, haystack_len)
                    ax.grid(True)
                    ax.minorticks_on()
                    ax.grid(which='major', linestyle='--', linewidth='0.75', color='gray')
                    # ax[step_count].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                    #set the grid to be on integer values for x-axis
                    ax.set_xticks(np.arange(-2, haystack_len, 1))
                    ax.set_yscale('log')
                    # ax[step_count].set_ylim(0.3, 3)
                    ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=12)

                step_count += 1
            key_count += 1

    if valA == "gaussA":
        for key in quartiles.keys():
            if "OLS" in key and "analytical" not in key:
                print(f"key_count: {key_count}")
                print(f"quartiles[{key}].shape: {quartiles[key].shape}")
                step_count = 0
                for step in steps_in:
                    y = quartiles[key][1, :, real_steps[step_count]]
                    y -= 1
                    ax[step_count].axhline(y[0], label=key[:3] + "-" + key[7:], color=colors[key_count], linewidth=2, linestyle='-')


                    ax[step_count].legend(fontsize = 10, ncol=5, columnspacing=0.4, handletextpad=0.25)#, loc="upper left")
                    ax[step_count].set_xlabel("Needle Position from the End of the Haystack", fontsize=12, fontname="Times New Roman")
                    ax[step_count].set_ylabel("Error" + (" Ratio" if valA == "gaussA" else "") + f": {step} After Open", fontsize=12)
                    ax[step_count].set_xlim(-3, haystack_len)
                    ax[step_count].grid(True)
                    ax[step_count].minorticks_on()
                    ax[step_count].grid(which='major', linestyle='--', linewidth='0.75', color='gray')
                    # ax[step_count].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                    #set the grid to be on integer values for x-axis
                    ax[step_count].set_xticks(np.arange(-2, haystack_len, 1))
                    ax[step_count].set_yscale('log')
                    # ax[step_count].set_ylim([2e-1, 3])
                    ax[step_count].tick_params(axis='x', which='both', labelbottom=True, labelsize=12)
                    step_count += 1
                key_count += 1


    fig.tight_layout()

    plt.show()

    #add the date and time to the filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    os.makedirs(f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_sys_trace/needle_in_haystack_examples/{datasource}", exist_ok=True)
    fig.savefig(f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_sys_trace/needle_in_haystack_examples/{datasource}/error_ratios_{valA}_state_dim_{state_dim}{valC}_step_{ckpt_step}_haystack_len_{haystack_len}_{timestamp}.pdf", transparent=True)

    return None


def plot_steps_after_open_token(haystack_len, quartiles, seg_ext_quartiles, colors, valA, experiment, datasource, open_paren_ind, n_positions, len_seg_haystack, nope):

    if valA == "gaussA":
        quartilez_npz = quartiles
        seg_ext_quartiles_npz = seg_ext_quartiles
        quartiles = {key: quartiles[key] for key in quartiles.keys()}
        seg_ext_quartiles = {key: seg_ext_quartiles_npz[key] for key in seg_ext_quartiles_npz.keys()}
        for key in quartiles.keys():
            seg_ext_quartiles[key] -= 1
            quartiles[key] -= 1


    #make a figure with haystack_len subplots
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 3.5))
    # fig, ax = plt.subplots(haystack_len, 1, sharex=True, figsize=(5, 5*haystack_len))

    col_count = 0
    dither = 0.05

    fin_seg_len = n_positions-open_paren_ind -1
    x_values = np.arange(1, fin_seg_len+1)

    print(f"haystack_len: {haystack_len}")
    needle = haystack_len - 1
    for key in quartiles.keys():
        if "OLS_analytical" not in key and "Simulation" not in key and key != "OLS_ir_2":
            ax.errorbar(x_values + col_count*dither, quartiles[key][1, needle][open_paren_ind+1:-1], yerr=[quartiles[key][1, needle][open_paren_ind+1:-1] - quartiles[key][0, needle][open_paren_ind+1:-1], quartiles[key][2, needle][open_paren_ind+1:-1] - quartiles[key][1, needle][open_paren_ind+1:-1]], fmt='o', label="TF: Final Segment" if key == "MOP" else (key[:3] + "-" + key[7:] if "OLS" in key else f"{key}"), capsize=2, zorder=haystack_len if key == "MOP" else 0, color=colors[col_count])
            col_count += 1

    needle = 0
    # for key in seg_ext_quartiles.keys():
    #     if "OLS_analytical" not in key:



    y_values = seg_ext_quartiles["MOP"][1, needle][open_paren_ind - 1:open_paren_ind - 1 + fin_seg_len]
    yerr_lower = seg_ext_quartiles["MOP"][1, needle][open_paren_ind - 1:open_paren_ind - 1 + fin_seg_len] - seg_ext_quartiles["MOP"][0, needle][open_paren_ind - 1:open_paren_ind - 1 + fin_seg_len]
    yerr_upper = seg_ext_quartiles["MOP"][2, needle][open_paren_ind - 1:open_paren_ind - 1 + fin_seg_len] - seg_ext_quartiles["MOP"][1, needle][open_paren_ind - 1:open_paren_ind - 1 + fin_seg_len]

    ax.errorbar(x_values + dither*col_count, y_values, yerr=[yerr_lower, yerr_upper], fmt='o', label=f"TF w/o Punc.", capsize=2, zorder=haystack_len if key == "MOP" else 0, color=colors[col_count])
    col_count += 1

    needle = 0 #get the first needle
    open_paren_ind = 1
    key = "MOP"

    fin_seg_len = len_seg_haystack

    ax.errorbar(x_values[:10] + -dither, quartiles[key][1, needle][open_paren_ind+1: open_paren_ind + 1 + fin_seg_len], yerr=[quartiles[key][1, needle][open_paren_ind+1: open_paren_ind + 1 + fin_seg_len] - quartiles[key][0, needle][open_paren_ind+1: open_paren_ind + 1 + fin_seg_len], quartiles[key][2, needle][open_paren_ind+1: open_paren_ind + 1 + fin_seg_len] - quartiles[key][1, needle][open_paren_ind+1: open_paren_ind + 1 + fin_seg_len]], fmt='o', label="TF: Needle 0", capsize=2, zorder=haystack_len if key == "MOP" else 0, color=colors[col_count])
    col_count += 1
            
    ax.legend(ncol=3 if valA == "gaussA" else 1, fontsize=8)
    ax.grid(which="both")
    # ax.set_xlim(left=230, right=seg_ext_quartiles[key].shape[-1] - 1)  # set the x axis limits haystack_len*12 + 2
    # ax.set_ylim(bottom=0, top=5.5)  # set the y axis limits

    # Optionally, customize major and minor ticks
    ax.minorticks_on()

    # Set minor vertical grid lines to be on intervals of 1
    # Set major ticks on every interval of 50
    ax.set_xticks(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 5))

    # Set minor vertical grid lines to be on intervals of 1
    ax.set_xticks(range(int(ax.get_xlim()[0]), int(ax.get_xlim()[1]) + 1, 1), minor=True)

    ax.tick_params(axis='both', which='major', length=7, width=1, labelsize=12)
    ax.tick_params(axis='both', which='minor', length=4, width=0.5, labelsize=0)
    ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=12)
    ax.grid(which='major', linestyle='-', linewidth=1)
    ax.grid(which='minor', linestyle='--', linewidth=0.5)
    ax.set_ylabel(f"Error" + (" Ratio" if valA == "gaussA" else ""), fontsize=14)
    ax.set_xlabel("Steps after the Open Token", fontsize=14)
    ax.set_yscale('log')
    # ax.set_title(f"Prediction Error for Needle Position {needle}", fontsize=30)
    # if valA == "gaussA":
    #     ax.set_ylim([2e-1, 3])

    #add the date and time to the filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    # #add a caption to the bottom of the figure
    # fig.text(0.5, 0.1, f"Median of {num_examples} haystack configuration examples. step=" + str(ckpt_step) + "_" + timestamp, ha='center', fontsize=30)
    plt.tight_layout()

    os.makedirs(f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_sys_trace/needle_in_haystack_examples/{datasource}", exist_ok=True)
    fig.savefig(f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_sys_trace/needle_in_haystack_examples/{datasource}/last_seg_context_{valA}_haystack_len_{haystack_len}_{timestamp}.pdf", transparent=True)
    return None


def compute_quartiles_ckpt(config, steps_in, model_dir, experiment, kal_ckpt, haystack_len, ckpt_steps, train_conv_fin_quartiles_file, train_conv_beg_quartiles_file, x_values_file):

    nope = not config.use_pos_emb
    batch_size = config.batch_size
    gpus = len(config.devices)

    kal_err = None

    pred_ckpts = []
    last_pred_ckpt = 0
    x_values = []
    ys = {}
    y_errs = {}
    fin_quartiles_ckpt = {}
    beg_quartiles_ckpt = {}

    if config.val_dataset_typ == "gaussA":
        rat = True
        errs_dir = model_dir + experiment + f"/prediction_errors{config.C_dist}_step={kal_ckpt}.ckpt"
        errs_loc = errs_dir + f"/train_conv_needle_haystack_len_{haystack_len}_{config.datasource}_" + f"{config.val_dataset_typ}_state_dim_{config.nx}_"

        with open(errs_loc + "err_lss_examples.pkl", "rb") as f:
                kal_ckpt_errs = pickle.load(f)

        kal_err = kal_ckpt_errs["Kalman_rem"]

    else:
        rat = False

    for ckpt_step in ckpt_steps:

        errs_dir = model_dir + experiment + f"/prediction_errors{config.C_dist}_step={ckpt_step}.ckpt"
        errs_loc = errs_dir + f"/train_conv_needle_haystack_len_{haystack_len}_{config.datasource}_{config.val_dataset_typ}_state_dim_{config.nx}_" 

        if os.path.exists(errs_loc + "err_lss_examples.pkl"):
            print(f"loading errors for ckpt_step: {ckpt_step}")

            if len(pred_ckpts) > 0:
                last_pred_ckpt = pred_ckpts[-1]
                
                if config.n_embd == 128 and not nope: #Vanilla models
                    if config.val_dataset_typ == "ortho": #THIS IS FOR VANILLA ORTHO ONLY
                        gpus = 2 #just for ortho case
                    
                    elif config.val_dataset_typ == "ident" and ckpt_step > 9600: #THIS IS FOR VANILLA IDENT ONLY
                        gpus = 4

            else:
                last_pred_ckpt = 0
                if config.val_dataset_typ == "ortho" and config.n_embd == 128 and not nope: #THIS IS FOR VANILLA ORTHO ONLY
                    gpus = 3

            with open(errs_loc + "err_lss_examples.pkl", "rb") as f:
                err_lss_examples = pickle.load(f)

            # if os.path.exists(seg_ext_errs_loc + "err_lss_examples.pkl"):
            #     with open(seg_ext_errs_loc + "err_lss_examples.pkl", "rb") as f:
            #         seg_ext_err_lss_examples = pickle.load(f)

            if len(pred_ckpts) == 0:
                #get seg_starts
                seg_starts_per_conf = get_seg_starts_per_config(experiment, config.val_dataset_typ, config.C_dist, config.nx, ckpt_step, print_seg_starts=True, nope=nope, needle=True, haystack_len=haystack_len, train_conv=True)

            quartiles = comp_quartiles(err_lss_examples, ratio=rat, train_conv=True, kal_err=kal_err)

            del err_lss_examples
            #clear cuda cache
            torch.cuda.empty_cache()
            gc.collect()

            print(f"batch_size: {batch_size}, gpus: {gpus}, ckpt_step: {ckpt_step}, last_pred_ckpt: {last_pred_ckpt}")
            if len(x_values) > 0:
                x_value = batch_size*gpus*(ckpt_step - last_pred_ckpt) + x_values[-1]
            else:
                x_value = batch_size*gpus*(ckpt_step - last_pred_ckpt)

            print(f"x_value: {x_value}")
            x_values.append(x_value)
            for needle in range(1):
                fin_seg_start = seg_starts_per_conf[needle][-1]
                beg_seg_start = seg_starts_per_conf[needle][0]
                for step in steps_in:
                    for key in ["MOP"]:
                        if "OLS_analytical" not in key and key not in  ["Zero", "Analytical_Simulation", "Kalman_rem", "Kalman", "Analytical_Kalman"]:
                            
                            y = quartiles[key][1, needle, fin_seg_start + step]
                            
                            y_err = [
                                [quartiles[key][1, needle, fin_seg_start + step] - quartiles[key][0, needle, fin_seg_start + step]],
                                [quartiles[key][2, needle, fin_seg_start + step] - quartiles[key][1, needle, fin_seg_start + step]]
                            ]

                            if needle == 0:
                                if len(pred_ckpts) == 0:
                                    if step == 1:
                                        ys[key] = {}
                                        y_errs[key] = {}
                                        fin_quartiles_ckpt[key] = {}
                                        beg_quartiles_ckpt[key] = {}

                                    ys[key][step] = [y]
                                    y_errs[key][step] = [y_err]
                                    fin_quartiles_ckpt[key][step] = [quartiles[key][:, needle, fin_seg_start + step]]
                                    beg_quartiles_ckpt[key][step] = [quartiles[key][:, needle, beg_seg_start + step]]

                                else:

                                    ys[key][step].append(y)
                                    y_errs[key][step].append(y_err)
                                    fin_quartiles_ckpt[key][step].append(quartiles[key][:, needle, fin_seg_start + step])
                                    beg_quartiles_ckpt[key][step].append(quartiles[key][:, needle, beg_seg_start + step])


            pred_ckpts.append(ckpt_step)

            if os.path.exists(errs_loc + "err_lss_examples.pkl"):
                if os.access(errs_loc + "err_lss_examples.pkl", os.W_OK):
                    os.remove(errs_loc + "err_lss_examples.pkl") #delete the err_lss_examples.pkl file
                else:
                    print(f"path: {errs_loc + "err_lss_examples"} for ckpt_step: {ckpt_step} is not writable.")
            else:
                print(f"path: {errs_loc + "err_lss_examples"} for ckpt_step: {ckpt_step} does not exist.")
        else:
            print(f"path: {errs_loc + "err_lss_examples"} for ckpt_step: {ckpt_step} does not exist.")


    os.makedirs(os.path.dirname(train_conv_fin_quartiles_file), exist_ok=True)
    #save quartiles to pickle file
    with open(train_conv_fin_quartiles_file, "wb") as f:
        pickle.dump(fin_quartiles_ckpt, f)

    os.makedirs(os.path.dirname(train_conv_beg_quartiles_file), exist_ok=True)
    #save quartiles to pickle file
    with open(train_conv_beg_quartiles_file, "wb") as f:
        pickle.dump(beg_quartiles_ckpt, f)

    os.makedirs(os.path.dirname(x_values_file), exist_ok=True)
    np.save(x_values_file, x_values)


    return fin_quartiles_ckpt, beg_quartiles_ckpt, x_values

def load_quartiles_ckpt_files(haystack_len, model_dir, experiment):
    train_conv_fin_quartiles_file = model_dir + experiment + f"/needles/train_conv/train_conv_fin_quartiles_haystack_len_{haystack_len}.pkl"
    train_conv_beg_quartiles_file = model_dir + experiment + f"/needles/train_conv/train_conv_beg_quartiles_haystack_len_{haystack_len}.pkl"
    x_values_file = model_dir + experiment + f"/needles/train_conv/x_values_haystack_len_{haystack_len}.npy"

    fin_quartiles_ckpt = None
    beg_quartiles_ckpt = None
    x_values = None
    


    if os.path.exists(train_conv_fin_quartiles_file):
        print(f"Loading train conv quartiles from {train_conv_fin_quartiles_file}")
        with open(train_conv_fin_quartiles_file, "rb") as f:
            fin_quartiles_ckpt = pickle.load(f)

    if os.path.exists(train_conv_beg_quartiles_file):
        print(f"Loading train conv quartiles from {train_conv_beg_quartiles_file}")
        with open(train_conv_beg_quartiles_file, "rb") as f:
            beg_quartiles_ckpt = pickle.load(f)

    if os.path.exists(x_values_file):
        x_values = np.load(x_values_file)

    return train_conv_fin_quartiles_file, train_conv_beg_quartiles_file, x_values_file, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values

def plot_haystack_train_conv(colors, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values, valA, haystack_len, experiment, steps, nope):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4.7))

    # if valA == "ortho":
    #     steps = [1,2,3,5,10]
    # else:
    #     steps = [1,2,3]

    print(f"\n\n in haystack train conv plot valA: {valA}")

    for key in fin_quartiles_ckpt.keys():
        if key == "MOP":
            col_count = 0
            for step in steps:

                key_lab = "TF" if key == "MOP" else key
                qs = np.array(fin_quartiles_ckpt[key][step])
                qs = np.transpose(qs)

                if valA == "gaussA":
                    qs -= 1

                #if key contains OLS then repeat the values in qs to be the length of x_values
                if "OLS" in key:
                    print(f"key: {key} qs shape: {qs.shape}")
                    qs = np.repeat(qs, len(x_values), axis=0)
                    print(f"qs shape after repeat: {qs.shape}")
                ax.plot(x_values, qs[1], label=f"{key_lab}: {step} after final", markersize=5, marker=".", zorder=5 if key == "MOP" else 0, color=colors[col_count], linewidth=2)
                if not valA == "gaussA":
                    print("plotting error bars for fin")
                    ax.fill_between(x_values, qs[0], qs[2], alpha=0.2, color=colors[col_count])

                beg_qs = np.array(beg_quartiles_ckpt[key][step])
                beg_qs = np.transpose(beg_qs)
                #set the color to the same as the fin quartiles
                color = ax.get_lines()[-1].get_color()
                ax.plot(x_values, beg_qs[1], label=f"{key_lab}: {step} after initial", markersize=5, marker="x", color=color, linestyle="--", linewidth=2)

                if not valA == "gaussA":
                    print("plotting error bars for beg")
                    ax.fill_between(x_values, beg_qs[0], beg_qs[2], alpha=0.2, color=color)

                col_count += 1


    ax.set_xlabel("# of Training Examples", fontsize=14)
    ax.set_ylabel(f"Error " + ("Ratio" if valA == "gaussA" else ""), fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, which="both")
    ax.legend(fontsize=10, ncol=2 if valA =="ident" else 1, loc="lower left")
    ax.set_xlim(x_values[0] - 1e3, x_values[-1] + 1e3)
    # ax.set_ylim([5e-2, 3e0])
    # ax.set_title(("Ortho" if valA == "ortho" else ("Gaussian" if valA == "gaussA" else "Identity")) + f" Haystack Length: {haystack_len} vs Training Examples")
    plt.tight_layout()

    #add the date and time to the filename
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    os.makedirs(f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_sys_trace", exist_ok=True)
    fig.savefig(f"../outputs/GPT2" + ("_NoPE" if nope else "") + f"/{experiment}/figures/multi_sys_trace/{valA}_train_conv_haystack_len_{haystack_len}_{timestamp}.pdf", transparent=True, format="pdf")

    plt.show()
    return None

def haystack_plots(config, haystack_len, output_dir, ckpt_step, kal_step, compute_more=False):

    colors = ['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']

    open_paren_ind = (config.len_seg_haystack + 2)*haystack_len + 1 #compute teh open paren index

    steps_in = [1,2,3,5,10]

    model_dir, experiment = split_path(output_dir)
    
    if haystack_len == 19:
        if ckpt_step is not None:
            quartiles_file, seg_ext_quartiles_file, quartiles, seg_ext_quartiles = load_quartiles(model_dir, experiment, valC=config.C_dist, ckpt_step=ckpt_step, valA=config.val_dataset_typ, state_dim=config.nx, datasource=config.datasource)

            if quartiles is None or seg_ext_quartiles is None or compute_more:
                #get the err_lss_examples
                errs_dir = model_dir + experiment + f"/prediction_errors{config.C_dist}_step={ckpt_step}.ckpt"
                errs_loc = errs_dir + f"/needle_haystack_len_{config.num_sys_haystack}_{config.datasource}_" + f"{config.val_dataset_typ}_state_dim_{config.nx}_"
                seg_ext_errs_loc = errs_dir + f"/needle_haystack_len_{config.num_sys_haystack}_{config.datasource}_fin_seg_ext_" + f"{config.val_dataset_typ}_state_dim_{config.nx}_"

                with open(errs_loc + "err_lss_examples.pkl", "rb") as f:
                    err_lss_examples = pickle.load(f)

                with open(seg_ext_errs_loc + "err_lss_examples.pkl", "rb") as f:
                    seg_ext_err_lss_examples = pickle.load(f)

                
                if config.val_dataset_typ == "gaussA":
                    rat = True
                else:
                    rat = False
                quartiles = comp_quartiles(err_lss_examples, ratio=rat)
                seg_ext_quartiles = comp_quartiles(seg_ext_err_lss_examples, ratio=rat)

                save_quartiles(quartiles_file, quartiles, seg_ext_quartiles_file, seg_ext_quartiles)

            #plot needle position
            plot_needle_position(experiment, config.datasource, config.nx, ckpt_step, config.val_dataset_typ, config.C_dist, haystack_len, steps_in, open_paren_ind, quartiles, seg_ext_quartiles, colors, not config.use_pos_emb)

            #plot steps after open token
            plot_steps_after_open_token(haystack_len, quartiles, seg_ext_quartiles, colors, config.val_dataset_typ, experiment, config.datasource, open_paren_ind, config.n_positions, config.len_seg_haystack, not config.use_pos_emb)

            print(f"open_paren_ind: {open_paren_ind}")

                
        else:
            raise ValueError("last ckpt_step is none for haystack_len 19")
        
    
    # load quartiles_ckpt_files
    train_conv_fin_quartiles_file, train_conv_beg_quartiles_file, x_values_file, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values = load_quartiles_ckpt_files(haystack_len, model_dir, experiment)

    if fin_quartiles_ckpt is None or beg_quartiles_ckpt is None or x_values is None or compute_more:
        last_ckpt_file = get_last_checkpoint(model_dir + experiment + "/checkpoints")
        last_ckpt = last_ckpt_file.split("=")[1].split(".")[0]
        last_ckpt = int(last_ckpt)

        print(f"config.train_int: {config.train_int}, last_ckpt: {last_ckpt}")

        ckpt_steps = gen_ckpt_steps(config.train_int, last_ckpt, config.train_int) #make sure to set the train_int for testing

        #compute quartiles for train conv
        fin_quartiles_ckpt, beg_quartiles_ckpt, x_values = compute_quartiles_ckpt(config, steps_in, model_dir, experiment, kal_step, haystack_len, ckpt_steps, train_conv_fin_quartiles_file, train_conv_beg_quartiles_file, x_values_file)


    #plot haystack train conv
    plot_haystack_train_conv(colors, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values, config.val_dataset_typ, haystack_len, experiment, steps_in, not config.use_pos_emb)

    # #delete big files
    # #delete the err_lss_examples for this test run
    # if os.path.exists(errs_loc + "err_lss_examples.pkl"):
    #     os.remove(errs_loc + "err_lss_examples.pkl")
    # #delete the seg_ext_err_lss_examples for this test run
    # if os.path.exists(seg_ext_errs_loc + "err_lss_examples.pkl"):
    #     os.remove(seg_ext_errs_loc + "err_lss_examples.pkl")


    return None