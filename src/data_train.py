from collect_data import collect_data
from models import GPT2
from core import Config
from train import train_gpt2
from core import setup_train
import os
from create_plots_with_zero_pred import create_plots, convergence_plots, load_preds
import argparse
import wandb
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from log_log_fit import loglogfit, loglinfit, loss, model_function, model_function_loglin, loglogfit_regularized, closed_form_loglin, plot_closed_form_loglin_err
from scipy.optimize import curve_fit, minimize
import sympy as sp
import pickle
from check_ecdf import get_empirical_cdf
import gc
import torch
import shutil
import time
import subprocess
import sys
from get_last_checkpoint import get_last_checkpoint, split_path, find_smallest_step_subdir
from haystack_plots import haystack_plots, load_quartiles_ckpt_files, haystack_plots_train_conv_full, haystack_plots_needle_full
from gen_pred_cktps import gen_pred_ckpts
from core.training import mem_suppress_ckpt_path

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
os.environ["WANDB_SILENT"] = "true"

def wandb_train(config, config_dict, model, ckpt_dir, train_mix_dist=False, train_mix_state_dim=False):

    # 🐝 1️⃣ Start a new run to track this script
    run = wandb.init(
        # Set the project where this run will be logged
        project="transformer_kalman_no_sweep",
        # Track hyperparameters and run metadata
        config=config_dict,
        settings=wandb.Settings(_disable_stats=False, _disable_meta=False)
    )
    train_time = train_gpt2(model, config, ckpt_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim) # train the model

    print("Time spent training (days:hours:minutes):", time.strftime("%D:%H:%M", time.gmtime(train_time)))

    # Finish the run
    wandb.finish()

    # # Get the path to the current run directory
    # run_dir = run.dir

    # # Delete the run directory
    # shutil.rmtree(run_dir)

    # print(f"Deleted wandb run directory: {run_dir}")

    return None

def preds_thread(config, ckpt_path, make_preds, resume_train, train_conv, logscale, tf, output_dir, ys=None, sim_objs=None, train_mix_dist=False, train_mix_state_dim=False, run_kf_ols=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"  # check if cuda is available

    # create prediction plots
    run_preds = make_preds # run the predictions evaluation
    run_deg_kf_test = False #run degenerate KF test
    excess = False #run the excess plots
    shade = True
    config.override("ckpt_path", ckpt_path)
    print("\n\n\n config.ckpt_path:", config.ckpt_path)
    print("testing config.nx:", config.nx)
    print("\n\n")

    #get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)
    #get the parent directory of the parent directory
    ckpt_dir = os.path.dirname(parent_dir)
    # instantiate gpt2 model

    if resume_train:
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb)
        
        wandb_train(config, config_dict, model, ckpt_dir, train_mix_dist, train_mix_state_dim)

    # print(f"config.use_pos_emb: {config.use_pos_emb}")
    # print(f"in preds_thread: config.ckpt_path: {config.ckpt_path}")

    # ckpt = torch.load(config.ckpt_path, map_location=device)
    # print("ckpt.keys():", ckpt.keys())
    # print("ckpt['state_dict'].keys():", ckpt['state_dict'].keys())
    # print("wpe weight in state_dict:", "_backbone.wpe.weight" in ckpt['state_dict'])
    # print("wpe weight in state_dict value:", ckpt['state_dict']["_backbone.wpe.weight"])

    # empty_model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
    #             n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
    # print("empty_model keys:", empty_model.state_dict().keys())
    # print("empty_model wpe weight:", empty_model.state_dict()["_backbone.wpe.weight"])

    model = GPT2.load_from_checkpoint(config.ckpt_path,
                                n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb, map_location=device, strict=True).eval().to(
    device)

    create_plots(config=config, model=model, run_preds=run_preds, run_deg_kf_test=run_deg_kf_test, excess=excess, num_systems=config.num_val_tasks, shade=shade, logscale=logscale, train_conv=train_conv, tf=tf, ys=ys, sim_objs=sim_objs, run_kf_ols=run_kf_ols, output_dir=output_dir)

    return run_preds, run_deg_kf_test, excess, shade



def plot_train_conv(t, ax, subtract, error_checkpoints_tuples, y_values, x_values, y_fit_loglog, y_fit_loglin, a_loglog, b_loglog, c_loglog, a_loglin, b_loglin, c_loglin, a_opt, b_opt, c_opt, ts, sys, kfnorm, olsnorm, yax, xax, rem):

    if subtract > 0:
        plot_label_mean = "Mean - s, s=%g" % subtract
        plot_label_loglog = "Fit y-s = e^b*x^a - s, a=%g, b=%g, c=%g, s=%g" % (a_loglog, b_loglog, c_loglog, subtract)
        plot_label_loglin = "Fit y-s = e^b*e^(x*a) - s, a=%g, b=%g, c=%g, s=%g" % (a_loglin, b_loglin, c_loglin, subtract)

    else:
        plot_label_mean = "Mean"
        plot_label_loglog = "Fit y = e^b*x^a + c, a=%g, b=%g, c=%g" % (a_loglog, b_loglog, c_loglog)
        plot_label_loglin = "Fit y = e^b*e^(x*a) + c, a=%g, b=%g, c=%g" % (a_loglin, b_loglin, c_loglin)

    ax[t][sys].plot(x_values, y_values - subtract, marker='o', label=plot_label_mean)

    ax[t][sys].plot(x_values, y_fit_loglog - subtract, label=plot_label_loglog)

    ax[t][sys].plot(x_values, y_fit_loglin - subtract, label=plot_label_loglin)

    # ax[t][sys].plot(x_values, fitted_y_values_opt - subtract, label="Regularized Fit y-s = e^b*x^a, a=%g, b=%g, c=%g, s=%g" % (a_opt, b_opt, c_opt, subtract))

    # Assuming the above prints confirm the lists are 1-dimensional
    y1 = [x[1][t][1] for x in error_checkpoints_tuples]
    y2 = [x[1][t][2] for x in error_checkpoints_tuples]
    x = np.arange(len(error_checkpoints_tuples))

    #remove the entries after rem of y1, y2, and x
    y1 = y1
    y2 = y2
    x = x

    ax[t][sys].fill_between(x_values, y1-subtract, y2-subtract, alpha=0.2)
    ax[t][sys].set_title("System " + str(sys) + ": t = " + str(ts[t]) + ("_KF_normalized" if kfnorm else ("_OLS_normalized" if olsnorm else "")))
    ax[t][sys].set_xlabel("Checkpoint Step")
    ax[t][sys].set_ylabel("Error")

    # # Apply the formatter to the x-axis
    # ax[t][sys].xaxis.set_major_formatter(formatter)
    # ax[t][sys].legend()

    # Rotate the x-axis labels
    ax[t][sys].tick_params(axis='x', labelrotation=45)  # Rotate labels to 45 degrees
    # Adjust the label size if necessary
    ax[t][sys].tick_params(axis='x', labelsize=10)  # Adjust label size to 10 or any suitable size

    x_label_values = [int(x[0]) for x in error_checkpoints_tuples]
    ax[t][sys].set_xticklabels(x_label_values, rotation=45, fontsize=10)  # Rotate labels for better fit

    if yax == "log":
        # set y-axis to log scale
        ax[t][sys].set_yscale('log')
    if xax == "log":
        # set x-axis to log scale
        ax[t][sys].set_xscale('log')

    # add a legend 
    ax[t][sys].legend()

    return ax

def save_figure(fig, config, kfnorm, olsnorm, yax, xax, subtracted, err=False, ratios=False, cdf=False, eval_start=None):
    
    fig.text(0.5, 0, "The error bars are 3*std.", ha='center', va='bottom', fontsize=12)
    # Adjust layout to make room for the rotated x-axis labels
    plt.tight_layout()
    #get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    #get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
    fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + config.C_dist + "_system_conv_checks" + ("_KF_normalized" if kfnorm else ("_OLS_normalized" if olsnorm else "")) + ("_subtracted" if subtracted else "") + ("_ylog" if yax == "log" else "") + ("_xlog" if xax == "log" else "") + ("_fit_err" if err else "") + ("_dummy_ratios" if ratios else "") + ("_cdf" if cdf else "") + ("_" + str(eval_start) if eval_start else "")+ ".png")
    return None

def old_train_conv_code():
    run_preds, run_deg_kf_test, excess, shade = preds_thread(make_preds, resume_train, train_conv)

    kal_errors = None
    #load the prediction errors from the step=40000 prediction_errors file
    num_systems = config.num_val_tasks
    config.override("ckpt_path", "../outputs/GPT2/240617_150023.5e81de_rotDiagA_gauss_C/checkpoints/step=192000.ckpt")
    err_lss_load, irreducible_error_load, fir_bounds, rnn_errors, rnn_an_errors = load_preds(run_deg_kf_test, excess, num_systems, config)
    print("len of irreducible_error_load", len(irreducible_error_load))

    if kfnorm:
        kal_errors = np.mean(err_lss_load["Kalman"], axis=1)
    elif olsnorm:
        kal_errors = np.mean(err_lss_load["OLS_ir_length3_orig"], axis=1)

    #for loop to iterate through all the checkpoints in the output directory
    output_dir = "../outputs/GPT2/240617_150023.5e81de_rotDiagA_gauss_C"
    fig, axs = plt.subplots(1, config.num_val_tasks, figsize=(200, 20))  # 1 row, val_tasks columns, with a figure size of 100x20 inches
    ts = [50, 100, 200]

    if make_preds or t_conv_plot:

        filecount = 0
        sys_error_checkpoints_tuples = []
        sys_error_an_checkpoints_tuples = []
        for filename in os.listdir(output_dir + "/checkpoints/"):
            filecount += 1
            print("filecount:", filecount)
            config.override("ckpt_path", output_dir + "/checkpoints/" + filename)
            print("\n\n\nckpt_path", config.ckpt_path)
            step_avg_tup, step_avg_an_tup = convergence_plots(filecount, config, run_preds, run_deg_kf_test, kfnorm, config.num_val_tasks, shade, fig, axs, ts, kal_errors) #create the convergence plots and return the step and average error tuple

            # print("step_avg_tup[1]", step_avg_tup[1])
            sys_error_checkpoints_tuples.append(step_avg_tup) #append the tuple to the list of tuples

            sys_error_an_checkpoints_tuples.append(step_avg_an_tup) #append the tuple to the list of tuples

        #create the train_conv directory
        os.makedirs(output_dir + "/train_conv", exist_ok=True)

        #save sys_error_checkpoints_tuples to a pickle file
        with open(output_dir + "/train_conv/sys_error_checkpoints_tuples.pkl", "wb") as f:
            pickle.dump(sys_error_checkpoints_tuples, f)

        #save sys_error_an_checkpoints_tuples to a pickle file
        with open(output_dir + "/train_conv/sys_error_an_checkpoints_tuples.pkl", "wb") as f:
            pickle.dump(sys_error_an_checkpoints_tuples, f)
    else:
        #load sys_error_checkpoints_tuples from a pickle file
        with open(output_dir + "/train_conv/sys_error_checkpoints_tuples.pkl", "rb") as f:
            sys_error_checkpoints_tuples = pickle.load(f)

        #load sys_error_an_checkpoints_tuples from a pickle file
        with open(output_dir + "/train_conv/sys_error_an_checkpoints_tuples.pkl", "rb") as f:
            sys_error_an_checkpoints_tuples = pickle.load(f)

    #plot the error_checkpoints_tuples
    print("\n\nPlotting error_checkpoints_tuples")
    #make a new figure
    fig, ax = plt.subplots(len(ts), config.num_val_tasks, figsize=(30, 20))

    fig2, ax2 = plt.subplots(len(ts), config.num_val_tasks, figsize=(30, 20))

    figc, axc = plt.subplots(config.num_val_tasks, 1, figsize=(10, 20))

    fig_err, ax_err = plt.subplots(len(ts), config.num_val_tasks, figsize=(300, 30))

    figc_an, axc_an = plt.subplots(len(ts), config.num_val_tasks, figsize=(30, 20))

    fig_err_rats, ax_err_rats = plt.subplots(len(ts), 1, figsize=(20, 40))

    fig_err_rats_cdf, ax_err_rats_cdf = plt.subplots(len(ts), 1, figsize=(20, 40))

    # set the axis scaling
    yax = "lin"
    xax = "lin"

    #initialize the error dictionary list
    err_dict_list = initialize_err_list(ts)



    for sys in range(config.num_val_tasks):
        # Filter bairand transform sys_error_checkpoints_tuples for the current system sys
        error_checkpoints_tuples = [(str(x[0]), x[1][sys]) for x in sys_error_checkpoints_tuples if isinstance(x[1], list) and len(x[1]) > sys]

        error_checkpoints_an_tuples = [(str(x[0]), x[1][sys]) for x in sys_error_an_checkpoints_tuples if isinstance(x[1], list) and len(x[1]) > sys]
        
        #sort the error_checkpoints_tuples by the step
        error_checkpoints_tuples = sorted(error_checkpoints_tuples, key=lambda x: int(x[0]))

        error_checkpoints_an_tuples = sorted(error_checkpoints_an_tuples, key=lambda x: int(x[0]))
    
        #make a plot for each value of t in ts for each system
        for t in range(len(ts)):
            #create an error dictionary with the key being the name of the fit and the value being an empty list

            # Ensure that the indices are valid before accessing them
            try:
                y_an_values = [x[1][t][0] for x in error_checkpoints_an_tuples]
            except IndexError as e:
                print(f"IndexError: {e}")
                print(f"Error occurred at t={t} with error_checkpoints_an_tuples={error_checkpoints_an_tuples}")
                raise

            x_values = [float(x[0]) for x in error_checkpoints_tuples]

            #set the y_values to be the error
            y_values = [x[1][t][0] for x in error_checkpoints_tuples]

            #analytical
            # y_an_values = [x[1][t][0] for x in error_checkpoints_an_tuples]
            
            #keep only the first rem elements of x_values and y_values
            rem = int(np.ceil(len(x_values)/2))
            eval_start = len(x_values) - 1 #set eval_start to the last element of x_values
            x_train = x_values[:rem]
            y_train = y_values[:rem]

            # #analytical
            # y_an_values = y_an_values[rem:]

            ##### create a helper function for log optimization #######################################
            # closed form solution for loglin fit
            axc, a_vals, b_vals, c_vals, err_vals, err_lin_vals = plot_closed_form_loglin_err(x_train, y_train, irreducible_error_load[sys], axc, sys, ts[t], 0.0, np.mean(y_train))


            # #analytical
            # axc_an, a_vals_an, b_vals_an, c_vals_an, err_vals_an, err_lin_vals_an = plot_closed_form_loglin_err(x_values, y_an_values, irreducible_error_load[sys], axc_an, sys, ts[t], 0.0, np.mean(y_an_values))

            # get index for minimum lin error
            min_err_lin_idx = np.argmin(err_lin_vals)

            # #analytical
            # min_err_lin_idx_an = np.argmin(err_lin_vals_an)

            #get min c value
            min_c = c_vals[min_err_lin_idx]
            interval = 7e-3
            axc, a_vals, b_vals, c_vals, err_vals, err_lin_vals = plot_closed_form_loglin_err(x_train, y_train, irreducible_error_load[sys], axc, sys, ts[t], min_c - interval, min_c + interval)

            # get index for minimum lin error
            min_err_lin_idx = np.argmin(err_lin_vals)
            
            #get fitted y values from model function
            yfit_optc = model_function_loglin(x_values, a_vals[min_err_lin_idx], b_vals[min_err_lin_idx], c_vals[min_err_lin_idx])
            ###########################################################################################

            #plot error
            ax_err, p, lstsq_mean_err = fit_curves_err(yfit_optc, y_values, x_values, rem, ax_err, "Least Squares Optimal c=%g, a=%g, b=%g" % (c_vals[min_err_lin_idx], a_vals[min_err_lin_idx], b_vals[min_err_lin_idx]), t, ts, sys, eval_start=eval_start)

            # #analytical
            # min_c_an = c_vals_an[min_err_lin_idx_an]
            # axc_an, a_vals_an, b_vals_an, c_vals_an, err_vals_an, err_lin_vals_an = plot_closed_form_loglin_err(x_values, y_an_values, irreducible_error_load[sys], axc_an, sys, ts[t], min_c_an - interval, min_c_an + interval)

            # #analytical
            # yfit_optc_an = model_function_loglin(x_values, a_vals_an[min_err_lin_idx_an], b_vals_an[min_err_lin_idx_an], c_vals_an[min_err_lin_idx_an])

            #initial guess for the parameters
            initial_guess = [a_vals[min_err_lin_idx], b_vals[min_err_lin_idx], c_vals[min_err_lin_idx]]

            # Fit a line to the data (line on log-log scale)
            y_fit_loglog, a_loglog, b_loglog, c_loglog = loglogfit(x_train, x_values, y_train, initial_guess)

            ax_err, p, loglog_mean_err = fit_curves_err(y_fit_loglog, y_values, x_values, rem, ax_err, "y = e^bx^a + c, c=%g, a=%g, b=%g" % (c_loglog, a_loglog, b_loglog), t, ts, sys, past_y_max=p, eval_start=eval_start)
            
            # Fit a line to the data (line on log-linear scale)
            y_fit_loglin, a_loglin, b_loglin, c_loglin = loglinfit(x_train, x_values, y_train, initial_guess)

            ax_err, p, loglin_mean_err = fit_curves_err(y_fit_loglin, y_values, x_values, rem, ax_err, "y = e^be^(ax) + c, c=%g, a=%g, b=%g" % (c_loglin, a_loglin, b_loglin), t, ts, sys, past_y_max=p, eval_start=eval_start)

            # Fit a regularized line to the data
            # Regularization strength
            lambda_reg = 1e-2
            a_opt, b_opt, c_opt = loglogfit_regularized(initial_guess, x_train, y_train, lambda_reg)

            # Generate y-values based on the optimized model
            fitted_y_values_opt = model_function(x_values, a_opt, b_opt, c_opt)

            # ax_err, p, loglogreg_mean_err = fit_curves_err(fitted_y_values_opt, y_values, x_values, rem, ax_err, "Regularized Fit y = e^bx^a, c=%g, a=%g, b=%g" % (c_opt, a_opt, b_opt), t, ts, sys, eval_start=eval_start)

            #dumb predictor
            last_val = y_train[-1]
            yfit_dumb = np.full(len(x_values), last_val)
            ax_err, p, dumb_mean_err = fit_curves_err(yfit_dumb, y_values, x_values, rem, ax_err, "Dumb Predictor", t, ts, sys, past_y_max=p, eval_start=eval_start)

            #divide the mean errors by the dumb mean error
            lstsq_mean_err = lstsq_mean_err/dumb_mean_err
            loglog_mean_err = loglog_mean_err/dumb_mean_err
            loglin_mean_err = loglin_mean_err/dumb_mean_err
            # loglogreg_mean_err = loglogreg_mean_err/dumb_mean_err
            dumb_mean_err = dumb_mean_err/dumb_mean_err

            # add lstsq_mean_err to the err_dict_list
            err_dict_list[t]["lstsq"].append(lstsq_mean_err)

            # add loglog_mean_err to the err_dict_list
            err_dict_list[t]["loglog"].append(loglog_mean_err)

            # add loglin_mean_err to the err_dict_list
            err_dict_list[t]["loglin"].append(loglin_mean_err)

            # # add loglogreg_mean_err to the err_dict_list
            # err_dict_list[t]["loglogreg"].append(loglogreg_mean_err)

            # add dumb_mean_err to err_dict_list
            err_dict_list[t]["dumb"].append(dumb_mean_err)

            subtract = c_loglog #c_vals[min_err_lin_idx]

            ax = plot_train_conv(t, ax, subtract, error_checkpoints_tuples, y_values, x_values, y_fit_loglog, y_fit_loglin, a_loglog, b_loglog, c_loglog, a_loglin, b_loglin, c_loglin, a_opt, b_opt, c_opt, ts, sys, kfnorm, olsnorm, yax=yax, xax=xax, rem=rem)

            #plot the optimal c value
            ax[t][sys].plot(x_values, yfit_optc-subtract, label="Least Squares Optimal c=%g, a=%g, b=%g" % (c_vals[min_err_lin_idx], a_vals[min_err_lin_idx], b_vals[min_err_lin_idx]), linestyle='--')
            
            # #analytical
            # ax[t][sys].plot(x_values, yfit_optc_an-subtract, label="Least Squares Optimal Analytical c=%g, a=%g, b=%g" % (c_vals_an[min_err_lin_idx], a_vals_an[min_err_lin_idx], b_vals_an[min_err_lin_idx]), linestyle='--')

            ax[t][sys].legend()

            ax2 = plot_train_conv(ax2, np.float64(0.0), error_checkpoints_tuples, y_values, x_values, y_fit_loglog, y_fit_loglin, a_loglog, b_loglog, c_loglog, a_loglin, b_loglin, c_loglin, a_opt, b_opt, c_opt, ts, sys, kfnorm, olsnorm, yax=yax, xax=xax, rem=rem)
            ax2[t][sys].plot(x_values, yfit_optc, label="Least Squares Optimal c=%g, a=%g, b=%g" % (c_vals[min_err_lin_idx], a_vals[min_err_lin_idx], b_vals[min_err_lin_idx]), linestyle='--')

            # #analytical
            # ax2[t][sys].plot(x_values, yfit_optc_an, label="Least Squares Optimal Analytical c=%g, a=%g, b=%g" % (c_vals[min_err_lin_idx], a_vals[min_err_lin_idx], b_vals[min_err_lin_idx]), linestyle='--')

            ax2[t][sys].legend()
            ax_err[t][sys].legend()
            fig_err.tight_layout()

    for t in range(len(ts)):
        if t == 0:
            #get indices of the sorted list 
            indices = np.argsort(err_dict_list[t]["loglin"]) #sort the loglin errors in ascending order and get the indices
            print("indices", indices)
            print("err_dict_list[t][loglin]", err_dict_list[t]["loglin"])

        
        #sort err_dict_list[t]["loglin"] by the indices and name it sorted_loglin
        sorted_loglin = np.array(err_dict_list[t]["loglin"])[indices]
        #sort err_dict_list[t]["loglog"] by the indices and name it sorted_loglog
        sorted_loglog = np.array(err_dict_list[t]["loglog"])[indices]
        #sort err_dict_list[t]["lstsq"] by the indices and name it sorted_lstsq
        sorted_lstsq = np.array(err_dict_list[t]["lstsq"])[indices]
        #sort err_dict_list[t]["dumb"] by the indices and name it sorted_dumb
        sorted_dumb = np.array(err_dict_list[t]["dumb"])[indices]
        #sort err_dict_list[t]["loglogreg"] by the indices and name it sorted_loglogreg
        # sorted_loglogreg = np.array(err_dict_list[t]["loglogreg"])[indices]

        #plot the error ratios
        ax_err_rats[t].scatter(np.arange(len(sorted_lstsq)), sorted_lstsq, label="Least Squares", s=200, marker='.')
        ax_err_rats[t].scatter(np.arange(len(sorted_loglog)), sorted_loglog, label="Log-Log", s=200, marker='.')
        ax_err_rats[t].scatter(np.arange(len(sorted_loglin)), sorted_loglin, label="Log-Lin", s=200, marker='.')
        # ax_err_rats[t].scatter(np.arange(len(sorted_loglogreg)), sorted_loglogreg, label="Log-Log Regularized", s=200, marker='.')
        ax_err_rats[t].scatter(np.arange(len(sorted_dumb)), sorted_dumb, label="Dumb", s=200, marker='.')
        ax_err_rats[t].set_title("Ratio of MSE over Dummy MSE: t = " + str(ts[t]))
        ax_err_rats[t].set_xlabel("System")
        ax_err_rats[t].set_ylabel("MSE Ratio")
        ax_err_rats[t].legend()

        #plot cdf of the error ratios
        ecdf_lstsq = get_empirical_cdf(err_dict_list[t]["lstsq"])
        ecdf_loglog = get_empirical_cdf(err_dict_list[t]["loglog"])
        ecdf_loglin = get_empirical_cdf(err_dict_list[t]["loglin"])
        # ecdf_loglogreg = get_empirical_cdf(err_dict_list[t]["loglogreg"])
        ecdf_dumb = get_empirical_cdf(err_dict_list[t]["dumb"])
        
        ax_err_rats_cdf[t].step(ecdf_lstsq.x, ecdf_lstsq.y, label="Least Squares", linewidth=2)
        ax_err_rats_cdf[t].step(ecdf_loglog.x, ecdf_loglog.y, label="Log-Log", linewidth=2)
        ax_err_rats_cdf[t].step(ecdf_loglin.x, ecdf_loglin.y, label="Log-Lin", linewidth=2)
        # ax_err_rats_cdf[t].step(ecdf_loglogreg.x, ecdf_loglogreg.y, label="Log-Log Regularized", linewidth=2)
        ax_err_rats_cdf[t].step(ecdf_dumb.x, ecdf_dumb.y, label="Dumb", linewidth=2)
        ax_err_rats_cdf[t].set_title("CDF of MSE Ratios: t = " + str(ts[t]))
        ax_err_rats_cdf[t].set_xlabel("MSE Ratio Value")
        # ax_err_rats_cdf[t].set_xlim([0, 1.25])
        ax_err_rats_cdf[t].set_ylabel("CDF")
        ax_err_rats_cdf[t].legend()


        #unsorted
        # #plot the error ratios
        # ax_err_rats[t].plot(np.arange(len(err_dict_list[t]["lstsq"])), err_dict_list[t]["lstsq"], label="Least Squares", linewidth=2, marker='.')
        # ax_err_rats[t].plot(np.arange(len(err_dict_list[t]["loglog"])), err_dict_list[t]["loglog"], label="Log-Log", linewidth=2, marker='.')
        # ax_err_rats[t].plot(np.arange(len(err_dict_list[t]["loglin"])), err_dict_list[t]["loglin"], label="Log-Lin", linewidth=2, marker='.')
        # # ax_err_rats[t].plot(np.arange(len(err_dict_list[t]["loglogreg"])), err_dict_list[t]["loglogreg"], label="Log-Log Regularized", linewidth=2, marker='.')
        # ax_err_rats[t].plot(np.arange(len(err_dict_list[t]["dumb"])), err_dict_list[t]["dumb"], label="Dumb", linewidth=2, marker='.')
        
    save_figure(fig, config, kfnorm, olsnorm, yax=yax, xax=xax, subtracted=True)
    save_figure(fig2, config, kfnorm, olsnorm, yax=yax, xax=xax, subtracted=False)
    save_figure_c(figc, config, kfnorm, olsnorm, yax=yax, xax=xax, subtracted=False)

    save_figure(fig_err, config, kfnorm, olsnorm, yax="lin", xax="lin", subtracted=False, err=True)
    save_figure(fig_err_rats, config, kfnorm, olsnorm, yax="lin", xax="lin", subtracted=False, err=False, ratios=True,eval_start=eval_start)
    save_figure(fig_err_rats_cdf, config, kfnorm, olsnorm, yax="lin", xax="lin", subtracted=False, err=False, ratios=True, cdf=True, eval_start=eval_start)

    # #analytical
    # save_figure_c(figc_an, config, kfnorm, olsnorm, yax=yax, xax=xax, subtracted=False)

    return None

def save_figure_c(fig, config, kfnorm, olsnorm, yax, xax, subtracted):
    
    fig.text(0.5, 0, "The error bars are 3*std.", ha='center', va='bottom', fontsize=12)
    # Adjust layout to make room for the rotated x-axis labels
    plt.tight_layout()
    #get the parent directory of the ckpt_path
    parent_dir = os.path.dirname(config.ckpt_path)

    #get the parent directory of the parent directory
    parent_parent_dir = os.path.dirname(parent_dir)
    os.makedirs(parent_parent_dir + "/figures", exist_ok=True)
    fig.savefig(parent_parent_dir + f"/figures/{config.dataset_typ}" + config.C_dist + "_find_opt_c" + ("_KF_normalized" if kfnorm else ("_OLS_normalized" if olsnorm else "")) + ("_subtracted" if subtracted else "") + ("_ylog" if yax == "log" else "") + ("_xlog" if xax == "log" else "") + ".png")
    return None

def get_opposite_color(hex_color):
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Calculate the complementary color
    comp_r = 255 - r
    comp_g = 255 - g
    comp_b = 255 - b

    # Convert the complementary RGB back to hex
    comp_hex = f'#{comp_r:02x}{comp_g:02x}{comp_b:02x}'

    return comp_hex

def fit_curves_err(fit_y, y_values, x_values, rem, ax_err, plot_label, t, ts, sys, eval_start=24, past_y_max=0):
    #compute the element-wise squared error between y_values and yfit_optc
    opt_err = (y_values - fit_y)**2

    # if eval_start != rem:
    #     raise ValueError("eval_start not to rem which is: ", rem)
    #compute the mean value of opt_err after the index rem
    mean_opt_err = np.mean(opt_err[eval_start:])

    #plot the error vs x_values on ax_err on a linear linear scale. Have the curve entries before and after rem be different colors
    ax_err[t][sys].plot(x_values, opt_err, label=plot_label + " t="+str(ts[t]), marker='.')

    #if plot label contains "Least Squares Optimal c", plot a vertical line at x = rem
    if "Least Squares Optimal c" in plot_label:
        #plot a vertical line at x = rem
        ax_err[t][sys].axvline(x=x_values[rem], color='r', linestyle='--', label="Train-Test Split")

    #set x and y labels
    ax_err[t][sys].set_xlabel("Checkpoint Step")
    ax_err[t][sys].set_ylabel("Squared Error")

    #set the title
    ax_err[t][sys].set_title("System " + str(sys) + ": t = " + str(ts[t]))

    #set the x-axis limits
    lower_x_limit = 50000
    upper_x_limit = x_values[-1]
    ax_err[t][sys].set_xlim([lower_x_limit, upper_x_limit])
    # Filter the data based on the new x-axis limits
    x_values = np.array(x_values)
    opt_err = np.array(opt_err)
    filtered_y = opt_err[(x_values >= lower_x_limit) & (x_values <= upper_x_limit)]
    if filtered_y.max() > past_y_max:
        ax_err[t][sys].set_ylim([0, filtered_y.max()])
        ax_err[t][sys].figure.canvas.draw()
        past_y_max = filtered_y.max()

    return ax_err, past_y_max, mean_opt_err

def initialize_err_list(ts):
    # create a list of dictionaries to store the errors for each value of t
    err_dict_list = [{"lstsq": [], "loglog": [], "loglin": [], "loglogreg": [], "dumb": []} for t in range(len(ts))]
    return err_dict_list

def gen_ckpt_pred_steps(model_name): #change this function to use the model name

    #generate specific ckpt steps to predict on
    #params for vanilla ident model:
    # if config.val_dataset_typ == "ident" and config.use_pos_emb and config.n_embd == 128:
    if model_name == "ident":
        minval = 100
        maxval = 17600
        train_int = 100
        phases = [600, 800, 9600, maxval]
        hande_code_scale = True

        ckpt_pred_steps = gen_pred_ckpts(minval,maxval, train_int, phases, hande_code_scale)

    #params for vanilla ortho model
    # elif config.val_dataset_typ == "ortho" and config.use_pos_emb and config.n_embd == 128:
    elif model_name == "ortho":

        ckpt_pred_steps = np.arange(3000, 105000, 3000)

    elif model_name == "ortho_haar_check":

        ckpt_pred_steps = np.arange(1, 3)

    elif model_name == "ortho_haar":
        minval = 1000
        maxval = 113000
        train_int = 1000

        phases = [minval, 10000, 22000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for vanilla gauss model:
    # elif config.val_dataset_typ == "gaussA" and config.use_pos_emb and config.n_embd == 128:
    elif model_name == "gauss":
        minval = 3000
        maxval = 180000
        train_int = 3000

        phases = [3000, 90000, 135000, maxval]
        hande_code_scale = False

        ckpt_pred_steps = gen_pred_ckpts(minval,maxval, train_int, phases, hande_code_scale)

    elif model_name == "gauss_zero_cut":
        minval = 1000
        maxval = 99000
        train_int = 1000

        phases = [minval, 10000, 19000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ident nope model:
    # elif config.val_dataset_typ == "ident" and not config.use_pos_emb:
    elif model_name == "ident_nope":
        minval = 100
        maxval = 15600
        train_int = 100

        phases = [100, 1100, 5600, 10600, maxval]
        hande_code_scale = False

        ckpt_pred_steps = gen_pred_ckpts(minval,maxval, train_int, phases, hande_code_scale)
    
    #params for ortho nope model
    # elif config.val_dataset_typ == "ortho" and not config.use_pos_emb:
    elif model_name == "ortho_nope":
        minval = 3000
        maxval = 201000
        train_int = 3000

        phases = [minval, 54000, 126000, maxval]
        hande_code_scale = False

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=hande_code_scale)

    #params for gaussian nope model:
    # elif config.val_dataset_typ == "gaussA" and not config.use_pos_emb:
    elif model_name == "gauss_nope":

        minval = 4000
        maxval = 216000
        train_int = 4000

        phases = [4000, 88000, 196000, maxval]
        hande_code_scale = False

        ckpt_pred_steps = gen_pred_ckpts(minval,maxval, train_int, phases, hande_code_scale)

    #params for gaussian tiny model:
    # elif config.val_dataset_typ == "gaussA" and config.n_embd == 72:
    elif model_name == "gauss_tiny":

        minval = 1000
        maxval = 180000
        train_int = 1000

        phases = [minval, 5000, 20000, 100000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho tiny model:
    # elif config.val_dataset_typ == "ortho" and config.n_embd == 72 and config.learning_rate > 3*1.584893192461114e-05 and config.acc_grad_batch == 1: 
    elif model_name == "ortho_tiny":
        minval = 1000
        maxval = 53000
        train_int = 1000

        phases = [minval, 24000, 48000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ident tiny model:
    # elif config.val_dataset_typ == "ident" and config.n_embd == 72:
    elif model_name == "ident_tiny":
        minval = 1000
        maxval = 54000
        train_int = 1000

        phases = [minval, 25000, 49000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ident big model:
    # elif config.val_dataset_typ == "ident" and config.n_embd == 192:
    elif model_name == "ident_big":
        minval = 100
        maxval = 149000
        train_int = 100

        phases = [minval, 200, 500, 5000, 10000, 20000, 50000, 68000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho big model:
    # elif config.val_dataset_typ == "ortho" and config.n_embd == 192:
    elif model_name == "ortho_big":

        minval = 5000
        maxval = 150000
        train_int = 5000
        ckpt_pred_steps = np.arange(minval, maxval, train_int)

    #params for gauss big model:
    # elif config.val_dataset_typ == "gaussA" and config.n_embd == 192:
    elif model_name == "gauss_big":
        minval = 100
        maxval = 147000
        train_int = 500

        phases = [2000, 5000, 50000, 82000, 98000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=True)

    #params for gauss small model:
    # elif config.val_dataset_typ == "gaussA" and config.n_embd == 96:
    elif model_name == "gauss_small":
        minval = 500
        maxval = 170000
        train_int = 500

        phases = [minval, 1000, 10000, 30000, 90000, 98500, 114000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ident small model:
    # elif config.val_dataset_typ == "ident" and config.n_embd == 96:
    elif model_name == "ident_small":
        minval = 50
        maxval = 40250
        train_int = 50

        phases = [minval, 550, 1000, 2000, 5000, 8400, 17700, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho small model:
    # elif config.val_dataset_typ == "ortho" and config.n_embd == 96:
    elif model_name == "ortho_small":

        minval = 500
        maxval = 172500
        train_int = 500

        phases = [minval, 4000, 10000, 22500, 66000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho tiny model single lr:
    # elif config.val_dataset_typ == "ortho" and config.n_embd == 72 and config.learning_rate > 0.95*1.584893192461114e-05  and config.learning_rate < 1.05*1.584893192461114e-05:
    elif model_name == "ortho_tiny_single_lr" or model_name == "ortho_tiny_single_lr_2" or model_name == "ortho_tiny_single_lr_3":
        minval = 1000
        maxval = 97000
        train_int = 1000

        phases = [minval, 10000, 21000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho tiny model double lr:
    # elif config.val_dataset_typ == "ortho" and config.n_embd == 72 and config.learning_rate > 1.95*1.584893192461114e-05  and config.learning_rate < 2.05*1.584893192461114e-05:
    elif model_name == "ortho_tiny_double_lr":
        minval = 1000
        maxval = 92000
        train_int = 1000

        phases = [minval, 10000, 21000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho tiny model smaller lr
    # elif config.val_dataset_typ == "ortho" and config.n_embd == 72 and config.learning_rate < 1.584893192461114e-05:
    elif model_name == "ortho_tiny_smaller_lr":

        minval = 1000
        maxval = 89000
        train_int = 1000

        phases = [minval, 10000, 19000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    #params for ortho tiny acc model:
    # elif config.val_dataset_typ == "ortho" and config.acc_grad_batch > 1:
    elif model_name == "ortho_tiny_acc":
        minval = 1000
        maxval = 53000
        train_int = 1000

        phases = [minval, 24000, 48000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
    
    #
    #
    elif model_name == "ortho_big_20k":
        minval = 1000
        maxval = 262000
        train_int = 1000

        phases = [minval, 3000, 27000, 50000, 100000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
    
    #
    #
    elif model_name == "ortho_big_20k":
        minval = 1000
        maxval = 262000
        train_int = 1000

        phases = [minval, 3000, 27000, 50000, 100000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_medium_single_gpu":
        minval = 1000
        maxval = 182000
        train_int = 1000

        phases = [minval, 8000, 30000, 80000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_big":
        minval = 1000
        maxval = 121000
        train_int = 1000

        phases = [minval, 10000, 52000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
        print(ckpt_pred_steps)
        print(len(ckpt_pred_steps))
    
    elif model_name == "ortho_haar_big_mask_backstory":
        minval = 33000
        maxval = 120000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)

    elif model_name == "ortho_haar_big_unmask_backstory":
        minval = 33000
        maxval = 57000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)



    elif model_name == "ortho_haar_medium_single_gpu":
        minval = 1000
        maxval = 182000
        train_int = 1000

        phases = [minval, 8000, 30000, 80000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_big":
        minval = 1000
        maxval = 121000
        train_int = 1000

        phases = [minval, 10000, 52000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
        print(ckpt_pred_steps)
        print(len(ckpt_pred_steps))
    
    elif model_name == "ortho_haar_big_mask_backstory":
        minval = 33000
        maxval = 120000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)

    elif model_name == "ortho_haar_big_unmask_backstory":
        minval = 33000
        maxval = 57000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)



    elif model_name == "ortho_big_4k":
        minval = 1000
        maxval = 262000
        train_int = 1000

        phases = [minval, 3000, 27000, 50000, 100000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)


    elif model_name == "ortho_big_20k_single_gpu":
        minval = 1000
        maxval = 115000
        train_int = 1000

        phases = [minval, 10000, 27000, 70000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
    
    #
    #
    elif model_name == "ortho_big_20k":
        minval = 1000
        maxval = 262000
        train_int = 1000

        phases = [minval, 3000, 27000, 50000, 100000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_medium_single_gpu":
        minval = 1000
        maxval = 182000
        train_int = 1000

        phases = [minval, 8000, 30000, 80000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_big":
        minval = 1000
        maxval = 121000
        train_int = 1000

        phases = [minval, 10000, 52000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
        print(ckpt_pred_steps)
        print(len(ckpt_pred_steps))
    
    elif model_name == "ortho_haar_big_mask_backstory":
        minval = 33000
        maxval = 120000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)

    elif model_name == "ortho_haar_big_unmask_backstory":
        minval = 33000
        maxval = 57000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)



    elif model_name == "ortho_haar_medium_single_gpu":
        minval = 1000
        maxval = 182000
        train_int = 1000

        phases = [minval, 8000, 30000, 80000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_big":
        minval = 1000
        maxval = 121000
        train_int = 1000

        phases = [minval, 10000, 52000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)
        print(ckpt_pred_steps)
        print(len(ckpt_pred_steps))
    
    elif model_name == "ortho_haar_big_mask_backstory":
        minval = 33000
        maxval = 120000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)

    elif model_name == "ortho_haar_big_unmask_backstory":
        minval = 33000
        maxval = 57000
        train_int = 3000

        ckpt_pred_steps = np.arange(minval, maxval + train_int, train_int)



    elif model_name == "ortho_big_4k":
        minval = 1000
        maxval = 262000
        train_int = 1000

        phases = [minval, 3000, 27000, 50000, 100000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)


    elif model_name == "ortho_big_20k_single_gpu":
        minval = 1000
        maxval = 115000
        train_int = 1000

        phases = [minval, 10000, 27000, 70000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "ortho_haar_medium_single_gpu":
        minval = 1000
        maxval = 182000
        train_int = 1000

        phases = [minval, 8000, 30000, 80000, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    elif model_name == "hyperparameter_sweep":
        minval = 1000
        maxval = 12000
        train_int = 1000

        phases = [minval, maxval]

        ckpt_pred_steps = gen_pred_ckpts(minval, maxval, train_int, phases, hande_code_scale=False)

    return ckpt_pred_steps


def predict_all_checkpoints(config, ckpt_dir, output_dir, logscale, ys, sim_objs, model_name):
        
    kal_step = None
    if config.needle_in_haystack:
        # num_sys_haystack = 4
        config.override("num_test_traces_configs", 1)
        # config.override("num_sys_haystack", num_sys_haystack)
        # config.override("len_seg_haystack", int(config.n_positions/(num_sys_haystack + 1)) - 2)
        # if config.num_sys_haystack == 1:
        #     num_haystack_examples = 50
        # else:
        # if abs_err:
        #     num_haystack_examples = 1
        # else:
        #     num_haystack_examples = 50
        # config.override("num_haystack_examples", num_haystack_examples)

    else:
        if not config.zero_cut:
            config.override("num_test_traces_configs", 1)
            config.override("single_system", True)
    filecount = 0

    
    ckpt_pred_steps = gen_ckpt_pred_steps(model_name) #generate specific ckpt steps to predict on

    run_kf_ols = True
    for filename in os.listdir(ckpt_dir + "/checkpoints/"):

        if filename.endswith(".ckpt") == False:
            continue
        else:
            filename_step = filename.split("=")[1].split(".")[0]
            filename_step = int(filename_step)

            if config.needle_in_haystack and filename_step not in ckpt_pred_steps:
                continue


            filecount += 1
            print("\nfilecount:", filecount)
            ckpt_path = ckpt_dir + "/checkpoints/" + filename
            # print("ckpt_path:", ckpt_path)
            run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds=True, resume_train=False, train_conv=True, logscale=logscale, tf=True, run_kf_ols=run_kf_ols, ys=ys, sim_objs=sim_objs, output_dir=output_dir)

            if run_kf_ols:
                # filename looks like "step=40000.ckpt" get the step number
                kal_step = filename_step

            run_kf_ols = False

    print(f"kal_step: {kal_step}")

    return kal_step


def set_config_params(config, model_name):

    output_dir = None

    # ortho_tiny_acc is the only one where acc_grad_batch not 1
    if model_name == "ortho_tiny_acc":
        config.override("acc_grad_batch", 2) #accumulate 8 gradient batches before updating the weights
    else:
        config.override("acc_grad_batch", 1)

    if model_name == "gauss":
        print("\n\nGAUSSIAN MEDIUM MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "gaussA")  # dataset type
        config.override("val_dataset_typ", "gaussA")  # validation dataset type
        config.override("C_dist", "_gauss_C")  # C distribution
        config.override("nx", 10)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})

        config = Config()  # create a config object

        # Training settings overrides
        config.override("devices", [2, 3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 3000)  # number of steps between logging
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 to check if it changes the speed of the training process
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings overrides
        config.override("model_type", "GPT2")  # model type
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension
        config.override("n_dims_out", 5)  # IMPORTANT TO KEEP THIS AT 5 FOR NOW

        experiment_name = "250114_202420.3c1184_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000"

    elif model_name == "gauss_zero_cut":
        print("\n\nGAUSSIAN MEDIUM ZERO CUT MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "gaussA")  # dataset type
        config.override("val_dataset_typ", "gaussA")  # validation dataset type
        config.override("C_dist", "_gauss_C")  # C distribution
        config.override("nx", 10)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})

        config = Config()  # create a config object

        # Training settings overrides
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 99000)  # number of training steps
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 to check if it changes the speed of the training process
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings overrides
        config.override("model_type", "GPT2")  # model type
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension
        config.override("n_dims_out", 5)  # IMPORTANT TO KEEP THIS AT 5 FOR NOW

        experiment_name = "250127_001511.3ac954_multi_sys_trace_zero_cut_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000"


    elif model_name == "ortho":
        print("\n\nORTHOGONAL MEDIUM MODEL\n\n")

        # Dataset settings overrides
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # dataset type
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # validation dataset type
        config.override("C_dist", "_ident_C")  # C distribution
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0, 2, 3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 3000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        experiment_name = "250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

    elif model_name == "ortho_haar":
        print("\n\nORTHOGONAL HAAR MEDIUM MODEL\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)
        config.override("num_val_tasks", 100)
        config.override("dataset_typ", "ortho_haar")
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")
        config.override("C_dist", "_ident_C")
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)

        # Training settings
        config.override("devices", [0, 1, 2])
        config.override("train_steps", 1008000)
        config.override("num_epochs", 1)
        config.override("train_int", 1000)
        config.override("use_true_len", False)
        config.override("batch_size", 512)
        config.override("acc_grad_batch", 1)
        config.override("train_data_workers", 128)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)

        # Model settings
        config.override("model_type", "GPT2")
        config.override("use_pos_emb", True)
        config.override("n_positions", 250)
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override(
            "n_dims_in",
            int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny
        )
        config.override("n_dims_out", 5)

        experiment_name = "250331_030338.010fdb_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        output_dir = f"../outputs/{config.model_type}/{experiment_name}"

        ckpt_dir = f"/data/shared/ICL_Kalman_Experiments/model_checkpoints/{config.model_type}/{experiment_name}"

    elif model_name == "ortho_haar_check":
        print("\n\nORTHOGONAL HAAR CHECK MEDIUM MODEL\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)
        config.override("num_val_tasks", 100)
        config.override("dataset_typ", "ortho_haar")
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")
        config.override("C_dist", "_ident_C")
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)

        # Training settings
        config.override("devices", [0, 1, 2])
        config.override("train_steps", 1008000)
        config.override("num_epochs", 1)
        config.override("train_int", 1)
        config.override("use_true_len", False)
        config.override("batch_size", 8)
        config.override("acc_grad_batch", 1)
        config.override("train_data_workers", 128)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)

        # Model settings
        config.override("model_type", "GPT2")
        config.override("use_pos_emb", True)
        config.override("n_positions", 250)
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override(
            "n_dims_in",
            int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny
        )
        config.override("n_dims_out", 5)

        experiment_name = "250403_174451.5c606a_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        output_dir = f"../outputs/{config.model_type}/{experiment_name}"

        ckpt_dir = f"/data/shared/ICL_Kalman_Experiments/model_checkpoints//{config.model_type}/{experiment_name}"

    elif model_name == "ident":
        print("\n\nIDENTITY MEDIUM MODEL\n\n")

        experiment_name = "250124_052617.8dd0f8_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"


        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0, 1])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 100)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "gauss_small":
        print("\n\nGAUSSIAN SMALL MODEL\n\n")

        experiment_name = "250125_103302.919337_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_3.169786384922228e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_gauss_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 10)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 500)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 4096)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 96)
        config.override("n_layer", 6)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ortho_small":
        print("\n\nORTHOGONAL SMALL MODEL\n\n")

        experiment_name = "250125_104123.f75c04_multi_sys_trace_ortho_state_dim_5_ident_C_lr_3.169786384922228e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 500)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 4096)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 96)
        config.override("n_layer", 6)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ident_small":
        print("\n\nIDENTITY SMALL MODEL\n\n")

        experiment_name = "250125_110549.80eba5_multi_sys_trace_ident_state_dim_5_ident_C_lr_3.169786384922228e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 50)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 4096)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 96)
        config.override("n_layer", 6)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "gauss_big":
        print("\n\nGAUSSIAN BIG MODEL\n\n")

        experiment_name = "250125_202437.caf35b_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.3207437987531975e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_gauss_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 10)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 50)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ortho_big":
        print("\n\nORTHO BIG MODEL\n\n")

        experiment_name = "250125_204545.a2cee4_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 5000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ident_big":
        print("\n\nIDENTITY BIG MODEL\n\n")

        experiment_name = "250125_210849.09203d_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 100)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml


    elif model_name == "gauss_nope":
        print("\n\nGAUSSIAN NoPE MODEL\n\n")

        output_dir = "../outputs/GPT2_NoPE/250125_092007.f34194_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000"
    
        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_gauss_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 10)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 4000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # 512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", False)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
    
    elif model_name == "ortho_nope":

        print("\n\nORTHOGONAL NoPE MODEL\n\n")
    
        output_dir = "../outputs/GPT2_NoPE/250124_190131.5710d5_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"
    
        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [2, 3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 3000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", False)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ident_nope":
        
        print("\n\nIDENTITY NoPE MODEL\n\n")
    
        output_dir = "../outputs/GPT2_NoPE/250123_214343.0d4e0b_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [2, 3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 100)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", False)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "gauss_tiny":
        print("\n\nGAUSSIAN TINY MODEL\n\n")

        experiment_name = "250128_022150.04b6bf_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_6.339572769844456e-05_num_train_sys_40000"
        
        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "gaussA")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_gauss_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 10)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 500)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 8192)  # 512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ortho_tiny":

        print("\n\nORTHOGONAL TINY MODEL\n\n")

        experiment_name = "250128_022331.067361_multi_sys_trace_ortho_state_dim_5_ident_C_lr_6.339572769844456e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 500)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 8192)  # 512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 4*1.584893192461114e-05) 

    elif model_name == "ident_tiny":

        print("\n\nIDENTITY TINY MODEL\n\n")

        experiment_name = "250128_022310.fc649a_multi_sys_trace_ident_state_dim_5_ident_C_lr_6.339572769844456e-05_num_train_sys_40000"

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [0])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 500)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 8192)  # 512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 64)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

    elif model_name == "ortho_tiny_double_lr":

        print("\n\nORTHOGONAL TINY DOUBLE LR MODEL\n\n")

        experiment_name = "250212_222339.54331c_multi_sys_trace_ortho_state_dim_5_ident_C_lr_3.169786384922228e-05_num_train_sys_40000"

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        config.override("devices", [2, 3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1

        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 2*1.584893192461114e-05)  

    elif model_name == "ortho_tiny_single_lr":
        experiment_name = "250212_222500.72ce84_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY SINGLE LR MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [0, 1])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 1.584893192461114e-05) 

    elif model_name == "ortho_tiny_single_lr_2":
        output_dir = "../outputs/GPT2/250227_002035.32ac0e_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY SINGLE LR 2 MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [0, 2])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 1.584893192461114e-05) 

    elif model_name == "ortho_tiny_single_lr_3":
        output_dir = "../outputs/GPT2/250227_223751.3324b0_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY SINGLE LR 3 MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [0,1,2,3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 1024)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 1.584893192461114e-05) 

    elif model_name == "ortho_tiny_single_lr_2":
        experiment_name = "250227_002035.32ac0e_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY SINGLE LR 2 MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [0, 2])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 1.584893192461114e-05) 

    elif model_name == "ortho_tiny_single_lr_3":
        experiment_name = "250227_223751.3324b0_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY SINGLE LR 3 MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [0,1,2,3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 1024)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 1.584893192461114e-05)

    elif model_name == "ortho_tiny_smaller_lr":

        experiment_name = "250215_185541.3091e0_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY SMALLER LR MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [1, 3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 0.833333*1.584893192461114e-05) 



    elif model_name == "ortho_tiny_acc":

        experiment_name = "250221_215216.337051_multi_sys_trace_ortho_state_dim_5_ident_C_lr_6.339572769844456e-05_num_train_sys_40000"

        print("\n\nORTHOGONAL TINY ACC MODEL\n\n")

        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        config.override("devices", [0, 2])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int", 1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 2048)  # usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 72)
        config.override("n_layer", 3)
        config.override("n_head", 6)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml

        config.override("learning_rate", 4*1.584893192461114e-05) 

    elif model_name == "ortho_haar_medium_single_gpu":
        print("\n\nORTHO HAAR MEDIUM MODEL 1 GPU\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", 1.584893192461114e-05)


        experiment_name = "250418_125901.8d6b22_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

    elif model_name == "ortho_haar_big":
        experiment_name = "250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000"

        print("\n\nORTHO HAAR BIG MODEL\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        #mem_suppress experiment settings
        config.override("mem_suppress", False) #run the memory suppression experiment
        config.override("masking", False) #run the masking training run
        config.override("cached_data", False) #use masked backstories
        config.override("backstory", False) #use masked backstories
        config.override("init_seg", False) #use masked initial segments
        config.override("backstory_len", config.ny + 2) #length of the backstory
        config.override("mask_budget", 10) #max # of systems that will be masked on first appearance (alpha)
        
        # Training settings
        config.override("devices", [2,3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 128)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 12)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", 0.833333*1.584893192461114e-05) 
    
    elif model_name == "ortho_big_4k":
        print("\n\nORTHO BIG MODEL 4K TRACES\n\n")

        output_dir = "../outputs/GPT2/250315_022855.4ce819_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_4000"

        # Dataset settings
        config.override("num_tasks", 4000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [0,1])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 128)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 12)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", 0.833333*1.584893192461114e-05)

    
    elif model_name == "ortho_big_20k_single_gpu":
        print("\n\nORTHO BIG MODEL 20K TRACES\n\n")

        output_dir = "../outputs/GPT2/250403_005725.507fce_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_20000"

        # Dataset settings
        config.override("num_tasks", 20000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 8*40)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 12)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", np.sqrt((len(config.devices) * config.batch_size)/512)*(0.833333333)*1.584893192461114e-05)

    elif model_name == "ortho_haar_big_mask_backstory":
        experiment_name = "250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000"

        print("\n\nORTHO HAAR BIG MASKED BACKSTORY MODEL\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        #mem_suppress experiment settings
        config.override("mem_suppress", True) #run the memory suppression experiment
        config.override("masking", True) #run the masking training run
        config.override("cached_data", False) #use masked backstories
        config.override("backstory", True) #use masked backstories
        config.override("init_seg", False) #use masked initial segments
        config.override("backstory_len", config.ny + 2) #length of the backstory
        config.override("mask_budget", 10) #max # of systems that will be masked on first appearance (alpha)
        
        # Training settings
        config.override("devices", [2,3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 8*40)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 12)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", np.sqrt((len(config.devices) * config.batch_size)/512)*(0.833333333)*1.584893192461114e-05)

    elif model_name == "ortho_haar_big_unmask_backstory":
        experiment_name = "250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000"

        print("\n\nORTHO HAAR BIG UNMASKED BACKSTORY MODEL\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting

        #mem_suppress experiment settings
        config.override("mem_suppress", True) #run the memory suppression experiment
        config.override("masking", False) #run the masking training run
        config.override("backstory", True) #use masked backstories
        config.override("cached_data", False) #use masked backstories
        config.override("init_seg", False) #use masked initial segments
        config.override("backstory_len", config.ny + 2) #length of the backstory
        config.override("mask_budget", 10) #max # of systems that will be masked on first appearance (alpha)
        
        # Training settings
        config.override("devices", [2,3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 8*40)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 192)
        config.override("n_layer", 24)
        config.override("n_head", 12)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", np.sqrt((len(config.devices) * config.batch_size)/512)*(0.833333333)*1.584893192461114e-05)
        config.override("learning_rate", 0.833333*1.584893192461114e-05)
    
    elif model_name == "ortho_haar_medium_single_gpu":
        print("\n\nORTHO HAAR MEDIUM MODEL 1 GPU\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ortho_haar")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        config.override("devices", [3])  # which GPU
        config.override("train_steps", 1008000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",1000)  # number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        config.override("learning_rate", 1.584893192461114e-05)


        experiment_name = "250418_125901.8d6b22_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        output_dir = f"../outputs/{config.model_type}/{experiment_name}"

        ckpt_dir = f"/data/shared/ICL_Kalman_Experiments/model_checkpoints/{config.model_type}/{experiment_name}"
    
    
    elif model_name == "hyperparameter_sweep":
        print("\n\nHyperparameter Sweep\n\n")

        # Dataset settings
        config.override("num_tasks", 40000)  # number of training systems
        config.override("num_val_tasks", 100)  # number of test systems
        config.override("dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho"
        config.override("max_cond_num", 100)
        config.override("distinct_cond_nums", 10)
        config.override("val_dataset_typ", "ident")  # "unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho"
        config.override("C_dist", "_ident_C")  # "_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
        config.override("nx", 5)
        config.override("ny", 5)
        config.override("n_noise", 1)
        config.override("num_traces", {"train": 1, "val": 1000})
        config.override("changing", False)  # used only for plotting
        
        # Training settings
        #config.override("devices", [3])  # leave comented out for hyperparamter sweep, it is set in config
        config.override("train_steps", 12000)  # number of training steps (27000x3 = 81000 effective single GPU iterations) (num_tasks*num_traces[train])/batch_size
        config.override("num_epochs", 1)  # minimum number of epochs to train for
        config.override("train_int",5)  #1000 number of steps between logging (train interval)
        config.override("use_true_len", False)  # Flag for a dataset length to be num_tasks
        config.override("batch_size", 512)  # 2048 #512 #usually 512 (~35GB) tune this to fit into GPU memory
        config.override("train_data_workers", 128)  # set to 1 (check if it changes the speed of the training process)
        config.override("test_batch_size", 256)
        config.override("test_data_workers", 1)  # keep at 1
        
        # Model settings
        config.override("model_type", "GPT2")  # "GPT2" #"transfoXL" #"olmo"
        config.override("use_pos_emb", True)  # use positional embeddings
        config.override("n_positions", 250)  # 500 for extended OLS #250 #context length
        config.override("n_embd", 128)
        config.override("n_layer", 12)
        config.override("n_head", 8)
        config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2) if config.multi_sys_trace else config.ny)  # input dimension is the observation dimension + special token parentheses + special start token + payload identifier
        config.override("n_dims_out", 5)  # (IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml
        
        #config.override("learning_rate", 1.584893192461114e-05)


        experiment_name = None#"250407_133748.f39da8_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

        output_dir = None #f"../outputs/{config.model_type}/{experiment_name}"

        ckpt_dir = None #f"/data/shared/ICL_Kalman_Experiments/model_checkpoints/{config.model_type}/{experiment_name}"


    else:
        raise ValueError("Model name not recognized. Please choose from the following: gauss, gauss_tiny, gauss_small, gauss_big, gauss_nope, ortho, ortho_tiny, ortho_small, ortho_big, ortho_nope, ident, ident_tiny, ident_small, ident_big, ident_nope")

    output_dir = f"../outputs/{config.model_type}/{experiment_name}"

    ckpt_dir = f"/data/shared/ICL_Kalman_Experiments/model_checkpoints/{config.model_type}/{experiment_name}"


    return output_dir, ckpt_dir, experiment_name

def get_entries(config, f):
    if ((not config.needle_in_haystack) or config.datasource == "val" or config.datasource == "train_systems"):
        num_traces = config.num_traces["val"]
        if config.datasource == "val":
            num_tasks = config.num_val_tasks
        else:
            num_tasks = config.num_tasks
    elif config.datasource == "train" or config.datasource == "backstory_train":
        num_traces = config.num_traces["train"]
        num_tasks = config.num_tasks
    else:
        raise ValueError(f"datasource {config.datasource} not recognized")
    samples = pickle.load(f)
    if config.late_start is not None:
        ys = np.stack(
            [entry["obs"] for entry in samples], axis=0
        ).reshape((num_tasks, num_traces, 251, config.ny)).astype(np.float32)
    else:
        ys = np.stack(
            [entry["obs"][:config.n_positions + 1] for entry in samples], axis=0
        ).reshape((num_tasks, num_traces, config.n_positions + 1, config.ny)).astype(np.float32)
    gc.collect()  # Start the garbage collector
    return ys

    
def get_test_data(config, experiment_name, num_haystack_ex=50):
    # load the validation data

    #for BLISS server
    path = "/data/shared/ICL_Kalman_Experiments/train_and_test_data"
    if (config.datasource == "val"):

        path = path + f"/{config.val_dataset_typ}/"

        print(f"getting test data from datasource {config.datasource}")

        # get the sim objs for the validation data
        with open(path + ("opposite_ortho_" if config.opposite_ortho else "") + f"val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}" +("_sync_ind_10" if config.val_dataset_typ == "ortho_sync" else "") + "_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        #set ys to be the validation data
        print(path + ("opposite_ortho_" if config.opposite_ortho else "") + f"val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl")
        with open(path + ("opposite_ortho_" if config.opposite_ortho else "") + f"val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}" +("_sync_ind_10" if config.val_dataset_typ == "ortho_sync" else "") + ".pkl", "rb") as f:
            # samples = pickle.load(f)
            # # for every 2000 entries in samples, get the observation values and append them to the ys list
            # ys = np.stack(
            #     [entry["obs"][:config.n_positions + 1] for entry in samples], axis=0
            # ).reshape((config.num_val_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)
            ys = get_entries(config, f)

            gc.collect()  # Start the garbage collector to free up memory

    elif config.datasource == "train":

        print(f"getting test data from datasource {config.datasource}")


        path = path + f"/{config.dataset_typ}/"

        max_num_sys = num_haystack_ex + config.max_sys_trace #max number of systems to use for testing

        #get the sim_objs for the training data
        with open (path + f"train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        #set ys to be the training data
        with open(path + f"train_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
            #get train traces
            # samples = pickle.load(f)
            # ys = np.stack(
            #     [entry["obs"][:config.n_positions + 1] for entry in samples], axis=0
            # ).reshape((config.num_tasks, config.num_traces["train"], config.n_positions + 1, config.ny)).astype(np.float32)
            ys = get_entries(config, f)
            gc.collect()  # Start the garbage collector
        
        print(f"len of sim_objs {len(sim_objs)}")
        sim_objs = sim_objs[:max_num_sys]
        print(f"len of sim_objs {len(sim_objs)}")

        print(f"shape of ys: {ys.shape}")
        ys = ys[:max_num_sys]
        print(f"shape of ys: {ys.shape}")
    

    elif config.datasource == "train_systems": #DEPRECATED

        print(f"getting test data from datasource {config.datasource}")


        path = path + f"/{config.dataset_typ}/"

        #get the sim_objs for the training data
        with open(path + f"train_{config.dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
            sim_objs = pickle.load(f)

        #generate traces from the training systems
        collect_data(config, output_dir, "val", False, False, False, sim_objs) 

        with open(path + f"{config.datasource}_val_specA_spec_C_state_dim_{config.nx}.pkl", "rb") as f:
            #get train traces
            # samples = pickle.load(f)
            # ys = np.stack(
            #     [entry["obs"][:config.n_positions + 1] for entry in samples], axis=0
            # ).reshape((config.num_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)
            ys = get_entries(config, f)
            gc.collect()  # Start the garbage collector

    else:
        raise ValueError("Datasource not recognized. Please choose from the following: val, train, train_systems")

    return ys, sim_objs

def get_kal_step(output_dir, model_name):
    #get kal_step
    ckpt_pred_steps = gen_ckpt_pred_steps(model_name)
    for filename in os.listdir(output_dir + "/checkpoints/"):
        filename_step = filename.split("=")[1].split(".")[0]
        filename_step = int(filename_step)

        if filename_step not in ckpt_pred_steps:
            continue

        kal_step = filename_step
        return kal_step
    
def plot_needles(config, num_sys, output_dir, model_dir, experiment, num_haystack_examples, steps_in, colors, pred_ckpt_step, make_preds, resume_train, logscale, tf, train_mix_dist, train_mix_state_dim, desktop, last_haystack_len=19):
    if num_sys == last_haystack_len:
        # if desktop:
        #     last_ckpt_step = maxval_dict[(config.val_dataset_typ, config.n_embd, config.use_pos_emb)]
        # else:
        #     last_ckpt = get_last_checkpoint(output_dir + "/checkpoints/")

        #     if last_ckpt is not None:
        #         last_ckpt_step = last_ckpt.split("=")[1].split(".")[0]
        #     else:
        #         raise ValueError("get_last_checkpoint returned None")
            
        #     print(f"last_ckpt_step: {last_ckpt_step}")

        #check for err_lss_examples at the last ckpt
        errs_dir = model_dir + experiment + f"/prediction_errors{config.C_dist}_step={pred_ckpt_step}.ckpt"
        errs_loc = errs_dir + f"/needle_haystack_len_{num_sys}_{config.datasource}_{config.val_dataset_typ}_state_dim_{config.nx}_" + ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("irrelevant_tokens_" if config.irrelevant_tokens else "") + ("same_tokens_" if config.same_tokens else "") + ("paren_swap_" if config.paren_swap else "")

        if not os.path.exists(errs_loc + "err_lss_examples.pkl") and not desktop:
            print(f"err_lss_examples.pkl does not exist for non train conv at early stop ckpt")
            make_preds = True

            ckpt_path = ckpt_dir + f"/checkpoints/step={pred_ckpt_step}.ckpt"

            ys, sim_objs = get_test_data(config, output_dir, num_haystack_examples)

            #run none train_conv
            config.override("num_test_traces_configs", num_sys)
            run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv=False, logscale=logscale, tf=tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, ys=ys, sim_objs=sim_objs, output_dir=output_dir)
            print("finished making predictions for non train conv at early stop ckpt")

            #run no punctuation final segment
            config.override("needle_final_seg_extended", True)

            run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv=False, logscale=logscale, tf=tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, ys=ys, sim_objs=sim_objs, output_dir=output_dir)
            make_preds = False
        
        print("making needle plots for haystack len:", num_sys)
        haystack_plots_needle_full(config, num_sys, output_dir, pred_ckpt_step, steps_in, colors, compute_more=make_preds)
    
    return make_preds
    


# main function

if __name__ == '__main__':
    wandb.login()
    # Create the parser
    parser = argparse.ArgumentParser(description='Run Predictions or not.')

    # Add the arguments
    parser.add_argument('--saved_preds', help='Boolean. Just plot the errors for a previously evaluated checkpoint', action='store_true')
    parser.add_argument('--make_preds', help='Boolean. Run predictions and plot the errors for a previously trained checkpoint', action='store_true')
    parser.add_argument('--tf', help='Boolean. Run predictions for just the transformer and plot the errors for a previously trained checkpoint', action='store_true')
    parser.add_argument('--resume_train', help='Boolean. Resume training from a specific checkpoint', action='store_true')
    parser.add_argument('--train_conv', help='Boolean. make predictions for all checkpoints', action='store_true')
    parser.add_argument('--multi_haystack', help='Boolean. Run predictions for multiple haystack lengths', action='store_true')
    parser.add_argument('--kfnorm', help='Boolean. subtract kalman performance from error', action='store_true')
    parser.add_argument('--olsnorm', help='Boolean. subtract kalman performance from error', action='store_true')
    parser.add_argument('--t_conv_plot', help='Boolean. plot the convergence plots with t as the indep. var.', action='store_true')
    parser.add_argument('--logscale', help='Boolean. plot the system test evaluations on a logscale with KF err subtracted.', action='store_true')
    parser.add_argument('--logscale_an', help='Boolean. plot the system test evaluations on a logscale divided by analytical KF err.', action='store_true')
    parser.add_argument('--train_mix_dist', help='Boolean. generate training data from a mixture of gaussian, uppertriA, and rotdiagA', action='store_true')
    parser.add_argument('--train_mix_state_dim', help='Boolean. generate training data from a mixture of state dimensions', action='store_true')
    parser.add_argument('--train_mix_C', help='Boolean. generate training data from a mixture of C dists', action='store_true')
    parser.add_argument('--part_train_set', help='Boolean. train on a subset of a previous experiments train dataset', action='store_true')
    parser.add_argument('--model_name', type=str, help='Name of the model to use')
    parser.add_argument('--abs_err', help='Boolean. Do not take the ratios of the gauss errors', action='store_true')
    parser.add_argument('--desktop', help='Boolean. Run on desktop', action='store_true')
    parser.add_argument('--datasource', type=str, help='Name of the datasource to use', default="val")
    parser.add_argument('--late_start', type=int, help="Integer. Start traces from a later index for interleaving at test time", default=None)
    parser.add_argument('--last_ckpt', help='Boolean. Take last checkpoint for needle plots', action='store_true')
    parser.add_argument('--zero_cut', help='Boolean. Run zero cut trace interleaving', action='store_true')
    parser.add_argument('--paren_swap', help='Boolean. Run experiment that swaps the open paren for the query in needle in a haystack tests', action='store_true')  
    parser.add_argument('--fix_needle', help='Boolean. Fix the needle in the haystack to be the same system for every example', action='store_true')
    parser.add_argument('--opposite_ortho', help='Boolean. generate training data from opposite orthogonal systems', action='store_true')
    parser.add_argument('--only_needle_pos', help='Boolean. only run the needle position evals', action='store_true')
    parser.add_argument('--same_tokens', help='Boolean. use the same special tokens for all systems in the haystack', action='store_true')
    parser.add_argument('--irrelevant_tokens', help='Boolean. use an irrelevant special token for query system in the haystack', action='store_true')
    parser.add_argument('--ortho_haar', help='Boolean. use orthogonal haar systems for test', action='store_true')
    parser.add_argument('--ortho', help='Boolean. use orthogonal systems test', action='store_true')
    parser.add_argument('--only_beg', help='Boolean. only run the beginning evals', action='store_true')
    parser.add_argument('--acc', help='Boolean. Using ACCESS server', action='store_true')
    parser.add_argument('--ortho_sync', help='Boolean. use orthogonal systems test with sync', action='store_true')

    parser.add_argument('--multi_train', help='Boolean. Run multiple runs back to back for hyperparameter sweep', action='store_true')
    parser.add_argument('--learning_rate', type=float, help='Override the default learning rate')




    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the flag
    print("saved preds arg", args.saved_preds)
    saved_preds = args.saved_preds
    print("make preds arg", args.make_preds)
    make_preds = args.make_preds
    print("tf arg", args.tf)
    tf = args.tf
    print("resume train arg", args.resume_train)
    resume_train = args.resume_train
    print("train conv arg", args.train_conv)
    train_conv = args.train_conv
    print("multi_haystack arg", args.multi_haystack)
    multi_haystack = args.multi_haystack
    print("kfnorm arg", args.kfnorm)
    kfnorm = args.kfnorm
    print("olsnorm arg", args.olsnorm)
    olsnorm = args.olsnorm
    print("t_conv_plot arg", args.t_conv_plot)
    t_conv_plot = args.t_conv_plot
    print("logscale arg", args.logscale)
    logscale = args.logscale
    print("logscale_an arg", args.logscale_an)
    logscale_an = args.logscale_an
    print("train_mix_dist arg", args.train_mix_dist)
    train_mix_dist = args.train_mix_dist
    print("train_mix_state_dim:", args.train_mix_state_dim)
    train_mix_state_dim = args.train_mix_state_dim
    print("train_mix_C:", args.train_mix_C)
    train_mix_C = args.train_mix_C
    print("part_train_set arg", args.part_train_set)
    part_train_set = args.part_train_set
    print("model_name arg", args.model_name)
    model_name = args.model_name
    print("abs arg", args.abs_err)
    abs_err = args.abs_err
    print("desktop arg", args.desktop)
    desktop = args.desktop
    print("datasource arg", args.datasource)
    datasource = args.datasource
    print("late_start arg", args.late_start)
    late_start = args.late_start
    print("last_ckpt arg", args.last_ckpt)
    last_ckpt = args.last_ckpt
    print("zero_cut arg", args.zero_cut)
    zero_cut = args.zero_cut
    print("paren_swap arg", args.paren_swap)
    paren_swap = args.paren_swap
    print("fix_needle arg", args.fix_needle)
    fix_needle = args.fix_needle
    print("opposite_ortho arg", args.opposite_ortho)
    opposite_ortho = args.opposite_ortho
    print("only_needle_pos arg", args.only_needle_pos)
    only_needle_pos = args.only_needle_pos
    print("same_tokens arg", args.same_tokens)
    same_tokens = args.same_tokens
    print("irrelevant_tokens arg", args.irrelevant_tokens)
    irrelevant_tokens = args.irrelevant_tokens
    print("ortho_haar arg", args.ortho_haar)
    ortho_haar = args.ortho_haar
    print("ortho arg", args.ortho)
    ortho = args.ortho
    print("only_beg arg", args.only_beg)
    only_beg = args.only_beg
    print("acc arg", args.acc)
    acc = args.acc
    print("ortho_sync arg", args.ortho_sync)
    ortho_sync = args.ortho_sync

    print("multi_train arg", args.multi_train)
    multi_train = args.multi_train
    print("learning_rate arg", args.learning_rate)
    new_learning_rate = args.learning_rate



    maxval_dict = {
    "ident": 17600,
    "ortho": 105000,
    "gauss": 180000,
    "ident_nope": 15600,
    "ortho_nope": 201000,
    "gauss_nope": 216000,
    "gauss_tiny": 180000,
    "ortho_tiny": 53000,
    "ident_tiny": 54000,
    "ident_big": 149000,
    "ortho_big": 150000,
    "gauss_big": 147000,
    "gauss_small": 170000,
    "ident_small": 40250,
    "ortho_small": 172500
}
#     kal_step_dict = { #currently inaccurate
#     (128, True): 126000,
#     (128, False): 15600,
#     (72, True): 180000,
#     (192, True): 149000,
#     (96, True): 170000
# }


    config = Config() # create a config object
    config.override("datasource", datasource) # set the datasource in the config object
    config.override("acc", acc) # set the acc in the config object for using the ACCESS server

    # config.override("late_start", late_start) # set the late_start in the config object
    config.override("late_start", late_start)

    config.override("paren_swap", paren_swap) # set the paren_swap in the config object
    if config.paren_swap:
        print("Running paren swap experiment\n\n\n")

    config.override("same_tokens", same_tokens) # set the same_tokens in the config object
    if config.same_tokens:
        print("Running same tokens experiment\n\n\n")

    config.override("irrelevant_tokens", irrelevant_tokens) # set the irrelevant_tokens in the config object
    if config.irrelevant_tokens:
        print("Running irrelevant tokens experiment\n\n\n")

    config.override("fix_needle", fix_needle) # set the fix_needle in the config object
    if config.fix_needle:
        print("Running fix needle experiment\n\n\n")

    config.override("opposite_ortho", opposite_ortho) # set the opposite_ortho in the config object
    if config.opposite_ortho:
        config.override("val_dataset_typ", "ortho")

    config.override("only_beg", only_beg) # set the only_beg in the config object
    if config.only_beg:
        print("only plotting the beginning evals\n\n\n")

    if zero_cut:
        config.override("multi_sys_trace", True)
        config.override("zero_cut", zero_cut)
        config.override("needle_in_haystack", False)

    
    # Get the class variables in dictionary format
    config_dict  = {
        "seed": 0,
        "fully_reproducible": False,
        "num_tasks": 1,
        "num_val_tasks": 1,
        "dataset_typ": "gaussA",
        "val_dataset_typ": "gaussA",
        "C_dist": "_gauss_C",
        "nx": 10,
        "ny": 5,
        "n_noise": 1,
        "num_traces": {"train": 40000, "val": 2000},
        "train_steps": 7,
        "batch_size": 28,
        "train_data_workers": 1,
        "test_batch_size": 2,
        "test_data_workers": 1,
        "num_epochs": 1,
        "n_positions": 250,
        "n_embd": 128,
        "n_layer": 12,
        "n_head": 8,
        "n_dims_in": 5,
        "n_dims_out": 5,
        "changing": False,
        "learning_rate": 3e-4,
        "weight_decay": 1e-2,
        "gradient_clip_algorithm": 'norm',
        "gradient_clip_val": 1.0
    }
    #change the num_tasks, num_val_tasks, dataset_typ, C_dist, nx, ny, n_noise, num_traces, train_steps, batch_size, train_data_workers, test_batch_size, test_data_workers, num_epochs, n_positions, n_embd, n_layer, n_head, n_dims_in, n_dims_out, changing, learning_rate, weight_decay, gradient_clip_algorithm, gradient_clip_val in config_dict to the values in config
    config_attributes = list(config_dict.keys())
    for key in config_attributes:
        config_dict[key] = config.__getattribute__(key)

    if (not (train_conv or multi_haystack)) and (make_preds or saved_preds or resume_train):

        output_dir, ckpt_dir, experiment_name = set_config_params(config, model_name)
        print(f"output_dir: {output_dir}")
        
        # ckpt_path = "/data/shared/ICL_Kalman_Experiments/model_checkpoints/GPT2/250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000/checkpoints/step=32000.ckpt" #normal training run

        # ckpt_path = "/data/shared/ICL_Kalman_Experiments/model_checkpoints/GPT2/backstory_masked_250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000/checkpoints/step=69000.ckpt" #masked_backstories
        checkpoint_step = 124000

        ckpt_path  = f"{ckpt_dir}/checkpoints/step={checkpoint_step}.ckpt"
        #output_dir = "../outputs/GPT2/250331_030338.010fdb_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"
        
        #"../outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/checkpoints/step=105000.ckpt"
        
        ckpt_path = "/data/shared/ICL_Kalman_Experiments/model_checkpoints/GPT2/backstory_unmasked_250501_221900.f583e5_multi_sys_trace_ortho_haar_state_dim_5_ident_C_lr_1.4766370475008905e-05_num_train_sys_40000/checkpoints/step=59000.ckpt" #unmasked_backstories
        
        run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv, logscale, tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, output_dir=output_dir, ys=None, sim_objs=None, run_kf_ols=False)
        
    elif train_conv or multi_haystack:

        kal_step = None
        last_haystack_len = 19

        if abs_err: #if we are not taking the ratios of the gauss errors
            num_haystack_examples = 1
        elif config.opposite_ortho:
            num_haystack_examples = 1
        else:
            if config.datasource == "train":
                num_haystack_examples = 500 #40000
            else:
                num_haystack_examples = 50 #number of haystack examples to use for testing

        config.override("num_haystack_examples", num_haystack_examples)

        output_dir, ckpt_dir, experiment_name = set_config_params(config, model_name)

        if config.mem_suppress:
            ckpt_dir = mem_suppress_ckpt_path(config, ckpt_dir)
            print(f"\n\nmem_suppress ckpt_dir: {ckpt_dir}")

            output_dir = mem_suppress_ckpt_path(config, output_dir, experiment_ind=16)
            print(f"\n\nmem_suppress output_dir: {output_dir}")

            experiment_name = mem_suppress_ckpt_path(config,experiment_name,experiment_ind=0)

            print(f"\n\nmem_suppress experiment_name: {experiment_name}")
        
            os.makedirs(output_dir, exist_ok=True)

        if ortho_sync: # run ortho sync test
            config.override("val_dataset_typ", "ortho_sync")

        if multi_haystack:
        
            if ortho_haar:
                config.override("val_dataset_typ", "ortho_haar")
            elif ortho:
                config.override("val_dataset_typ", "ortho")

            ckpt_pred_steps = gen_ckpt_pred_steps(model_name)

            # steps_in = [1,2,3,5,10]
            steps_in = list(range(1,9))

            colors=['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']
        
            if config.paren_swap:
                if fix_needle or opposite_ortho:
                    num_sys_haystacks = [2] #only run for 2 systems in the haystack for the fixed needle paren swap experiment
                else:
                    # num_sys_haystacks = [2] #only run for 2 systems in the haystack for the paren swap experiment
                    num_sys_haystacks = list(range(2,last_haystack_len+1))
            elif config.same_tokens or config.irrelevant_tokens:
                num_sys_haystacks = list(range(2,5))
                
            else:
                num_sys_haystacks = list(range(1,last_haystack_len+1))

            print("num_sys_haystacks:", num_sys_haystacks)

            config.override("needle_in_haystack", True)
            
            for num_sys in num_sys_haystacks:

                config.override("num_sys_haystack", num_sys)
                config.override("n_positions", (config.len_seg_haystack + 2)*(num_sys+1))
                
                if not make_preds:

                    print("not making predictions")

                    model_dir, experiment = split_path(output_dir)

                    # load quartiles_ckpt_files
                    train_conv_fin_quartiles_file, train_conv_beg_quartiles_file, x_values_file, fin_quartiles_ckpt, beg_quartiles_ckpt, x_values = load_quartiles_ckpt_files(config, num_sys, model_dir, experiment)

                    if fin_quartiles_ckpt is not None and beg_quartiles_ckpt is not None and x_values is not None:
                        print(f"quartiles already exist for haystack length {num_sys}")

                        if saved_preds:
                            print("saved preds")
                            
                            if config.val_dataset_typ == "gaussA" and not desktop:
                                kal_step = get_kal_step(output_dir, model_name)
                            
                            print("making train_conv plots for haystack len:", num_sys)
                            pred_ckpt_step = haystack_plots_train_conv_full(config, model_name, num_sys, output_dir, ckpt_dir, experiment_name, ckpt_pred_steps, kal_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)
                            print(f"pred_ckpt_step: {pred_ckpt_step}")

                            if config.val_dataset_typ == "ident" or last_ckpt: #choose last ckpt for identity system because overfitting phenomenon is not observed
                                if desktop:
                                    pred_ckpt_step = maxval_dict[model_name]
                                else:
                                    pred_ckpt_step = int(get_last_checkpoint(output_dir + "/checkpoints/").split("=")[1].split(".")[0])                            

                            print(f"pred_ckpt_step: {pred_ckpt_step}")
                            print(f"pred_ckpt_step: {pred_ckpt_step}")
                            if config.datasource == "val":
                                make_preds = plot_needles(config, num_sys, output_dir, model_dir, experiment, num_haystack_examples, steps_in, colors, pred_ckpt_step, make_preds, resume_train, logscale, tf, train_mix_dist, train_mix_state_dim, desktop, last_haystack_len) 

                            # print("\n\nmaking plots for haystack len:", num_sys)
                            # haystack_plots(config, num_sys, output_dir, ckpt_pred_steps, kal_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)
                        continue
                    else:
                        print(f"\n\nchecking for err_lss_examples")

                        ckpt_step_dir = find_smallest_step_subdir(output_dir)
                        if ckpt_step_dir is not None:
                            ckpt_step = ckpt_step_dir.split("=")[1].split(".")[0]
                        else:
                            print("ckpt_step_dir is None")
                            ckpt_step = config.train_int

                        print(f"ckpt_step: {ckpt_step}")


                        errs_dir = model_dir + experiment + f"/prediction_errors{config.C_dist}_step={ckpt_step}.ckpt"
                        errs_loc = errs_dir + f"/train_conv_needle_haystack_len_{num_sys}_{config.datasource}_{config.val_dataset_typ}_state_dim_{config.nx}_" + ("fix_needle_" if config.fix_needle else "") + ("opposite_ortho_" if config.opposite_ortho else "") + ("irrelevant_tokens_" if config.irrelevant_tokens else "") + ("same_tokens_" if config.same_tokens else "") + ("paren_swap_" if config.paren_swap else "")


                        if os.path.exists(errs_loc + "err_lss_examples.pkl"):
                            print(f"err_lss_examples.pkl exists for haystack length {num_sys}")

                            kal_step = get_kal_step(output_dir, model_name)

                            print("making train_conv plots for haystack len:", num_sys)
                            pred_ckpt_step = haystack_plots_train_conv_full(config, model_name, num_sys, output_dir, ckpt_dir, experiment_name, ckpt_pred_steps, kal_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)
                            print(f"pred_ckpt_step: {pred_ckpt_step}")

                            if config.val_dataset_typ == "ident" or last_ckpt: #choose last ckpt for identity system because overfitting phenomenon is not observed
                                if desktop:
                                    pred_ckpt_step = maxval_dict[model_name]
                                else:
                                    pred_ckpt_step = int(get_last_checkpoint(output_dir + "/checkpoints/").split("=")[1].split(".")[0]) 

                            if config.datasource == "val":
                                make_preds = plot_needles(config, num_sys, output_dir, model_dir, experiment, num_haystack_examples, steps_in, colors, pred_ckpt_step, make_preds, resume_train, logscale, tf, train_mix_dist, train_mix_state_dim, desktop, last_haystack_len)   
                            # if num_sys == 19:
                            #     # if desktop:
                            #     #     last_ckpt_step = maxval_dict[(config.val_dataset_typ, config.n_embd, config.use_pos_emb)]
                            #     # else:
                            #     #     last_ckpt = get_last_checkpoint(output_dir + "/checkpoints/")

                            #     #     if last_ckpt is not None:
                            #     #         last_ckpt_step = last_ckpt.split("=")[1].split(".")[0]
                            #     #     else:
                            #     #         raise ValueError("get_last_checkpoint returned None")
                                    
                            #     #     print(f"last_ckpt_step: {last_ckpt_step}")

                            #     #check for err_lss_examples at the last ckpt
                            #     errs_dir = model_dir + experiment + f"/prediction_errors{config.C_dist}_step={pred_ckpt_step}.ckpt"
                            #     errs_loc = errs_dir + f"/needle_haystack_len_{num_sys}_{config.datasource}_{config.val_dataset_typ}_state_dim_{config.nx}_"

                            #     if not os.path.exists(errs_loc + "err_lss_examples.pkl"):
                            #         print(f"err_lss_examples.pkl does not exist for non train conv at last ckpt")
                            #         make_preds = True

                            #         ckpt_path = ckpt_dir + f"/checkpoints/step={pred_ckpt_step}.ckpt"

                            #         ys, sim_objs = get_test_data(config, output_dir, num_haystack_examples)

                            #         #run none train_conv
                            #         config.override("num_test_traces_configs", num_sys)
                            #         run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv=False, logscale=logscale, tf=tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, ys=ys, sim_objs=sim_objs)
                            #         print("finished making predictions for non train conv at early stop ckpt")

                            #         #run no punctuation final segment
                            #         config.override("needle_final_seg_extended", True)

                            #         run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv=False, logscale=logscale, tf=tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, ys=ys, sim_objs=sim_objs)
                            #         make_preds = False
                                
                            #     print("making needle plots for haystack len:", num_sys)
                            #     haystack_plots_needle_full(config, num_sys, output_dir, pred_ckpt_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)
                            
                            continue

                print("\n\n\nstarting predictions for haystack len:", num_sys)
                start = time.time()

                config.override("needle_final_seg_extended", False)

                if opposite_ortho:
                    config.override("num_val_tasks", 2)

                ys, sim_objs = get_test_data(config, output_dir, num_haystack_examples)

                print(f"output dir: {output_dir}")
                print(f"config.use_pos_emb: {config.use_pos_emb}")

                print(f"shape of ys before predict_all_checkpoints: {ys.shape}")

                if not only_needle_pos:
                    #run train_conv

                    print(f"config.use_pos_emb: {config.use_pos_emb}")
                    kal_step = predict_all_checkpoints(config, ckpt_dir, output_dir, logscale, ys, sim_objs, model_name)

                    print(f"plotting train_conv convergence plots for haystack len {num_sys}")
                    pred_ckpt_step = haystack_plots_train_conv_full(config, model_name, num_sys, output_dir, ckpt_dir, experiment_name, ckpt_pred_steps, kal_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)

                    if config.val_dataset_typ == "ident" or last_ckpt: #choose last ckpt for identity system because overfitting phenomenon is not observed
                        if desktop:
                            pred_ckpt_step = maxval_dict[model_name]
                        else:
                            pred_ckpt_step = int(get_last_checkpoint(ckpt_dir + "/checkpoints/").split("=")[1].split(".")[0])

                if (num_sys == 19 and not abs_err) or paren_swap:
                    # #get the last checkpoint
                    # last_ckpt = get_last_checkpoint(output_dir + "/checkpoints/")

                    if paren_swap:
                        pred_ckpt_step = 27000

                    if pred_ckpt_step is not None:

                        ckpt_path = ckpt_dir + "/checkpoints/step=" + str(pred_ckpt_step) + ".ckpt"

                        print(f"non train conv config.num_haystack_examples: {config.num_haystack_examples}")

                        #run non train_conv
                        config.override("num_test_traces_configs", num_sys)
                        run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv=False, logscale=logscale, tf=tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, ys=ys, sim_objs=sim_objs, output_dir=output_dir)


                        #run no punctuation final segment
                        config.override("needle_final_seg_extended", True)

                        run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv=False, logscale=logscale, tf=tf, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, ys=ys, sim_objs=sim_objs, output_dir=output_dir)
                
                        # if last_ckpt is not None:
                        #     last_ckpt_step = last_ckpt.split("=")[1].split(".")[0]
                        # else:
                        #     last_ckpt_step = None

                        print("making needle plots for haystack len:", num_sys)
                        haystack_plots_needle_full(config, num_sys, output_dir, pred_ckpt_step, steps_in, colors, compute_more=make_preds)

                    else:
                        raise ValueError("pred_ckpt_step returned None")
                
                end = time.time()
                print(f"time elapsed for haystack len {num_sys} predictions (min): {(end - start)/60}")
                
                # haystack_plots(config, num_sys, output_dir, pred_ckpt_step, kal_step, compute_more=make_preds, abs_err=abs_err)
        else:

            output_dir = "../outputs/GPT2_NoPE/250123_214343.0d4e0b_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"
            if make_preds:

                ys, sim_objs = get_test_data(config, output_dir, num_haystack_examples)

                predict_all_checkpoints(config, ckpt_dir, output_dir, logscale, ys, sim_objs, model_name)
    
    elif new_learning_rate:

        #Multi-train parameters
        train_steps = 12000  #set this in set_config_params as well
        train_int = 1000
        model_name = "hyperparameter_sweep" #check this when running


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\ndevice:", device)

        
        print(f"\n\nstarting training for learning rate = {new_learning_rate}")

        
        config.override("train_steps", train_steps)
        config_dict["train_steps"] = train_steps
        config.override("train_int", train_int)
        config_dict["train_int"] = train_int
        config.override("learning_rate", new_learning_rate)
        config_dict["learning_rate"] = new_learning_rate

        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb)
    
        model.to(device)
        
        model.learning_rate = config.learning_rate

        output_dir, ckpt_dir, experiment_name = setup_train(model, train_mix_dist, train_mix_state_dim)
        
        
        # replace ckpt_path with the path to the checkpoint file
        config.override("ckpt_path", ckpt_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")

        wandb_train(config_dict, model, ckpt_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim) # train the model



        #### perfomrm predictions for one needle ####
        num_sys = 1
        num_haystack_examples = 50

        config.override("num_haystack_examples", num_haystack_examples)
        config.override("num_sys_haystack", num_sys)
        
        _, _, _ = set_config_params(config, model_name) #Not sure if this matters
        steps_in = list(range(1,11))

        if ortho_haar:
            config.override("val_dataset_typ", "ortho_haar")
        
        

        ckpt_pred_steps = gen_ckpt_pred_steps(model_name)

        colors=['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']
        config.override("needle_in_haystack", True)
        config.override("needle_final_seg_extended", False)

        ys, sim_objs = get_test_data(config, output_dir, num_haystack_examples)

        print(f"output dir: {output_dir}")
        print(f"config.use_pos_emb: {config.use_pos_emb}")

        print(f"shape of ys before predict_all_checkpoints: {ys.shape}")

        kal_step = predict_all_checkpoints(config, ckpt_dir, output_dir, logscale, ys, sim_objs, model_name)

        print(f"plotting train_conv convergence plots for haystack len {num_sys}")
        pred_ckpt_step = haystack_plots_train_conv_full(config, model_name, num_sys, output_dir, ckpt_dir, experiment_name, ckpt_pred_steps, kal_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)


    elif multi_train:
        #currently set for sweeping over learning rates
        sweep = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7] #1e-1, 1e-2,

        for lr in sweep:
            command = [
                sys.executable,  # This uses the current python interpreter
                "data_train.py",
                f"--learning_rate={lr}"
            ]
            subprocess.run(command)


        '''
        #Multi-train parameters
        train_steps = 20000  #set this in set_config_params as well
        train_int = 1000
        parameter = "learning_rate"
        sweep = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
        model_name = "hyperparameter_sweep" #check this when running


        print(f"performing sweep over {parameter} with values {sweep}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\ndevice:", device)

        for value in sweep:
            print(f"\n\nstarting training for {parameter} = {value}")



            #Build new config object for each run
            config = Config() # create a config object

            


            if zero_cut:
                config.override("multi_sys_trace", True)
                config.override("zero_cut", zero_cut)
                config.override("needle_in_haystack", False)

            config_attributes = list(config_dict.keys())
            for key in config_attributes:
                config_dict[key] = config.__getattribute__(key)
            
            config.override("train_steps", train_steps)
            config_dict["train_steps"] = train_steps
            config.override("train_int", train_int)
            config_dict["train_int"] = train_int
            config.override(parameter, value)
            config_dict[parameter] = value

            model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                    n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb)
        
            model.to(device)
        

            config.override("ckpt_path", None)
            config_dict["ckpt_path"] = None

            
            output_dir, ckpt_dir, experiment_name = setup_train(model, train_mix_dist, train_mix_state_dim)
            
            
            # replace ckpt_path with the path to the checkpoint file
            config.override("ckpt_path", ckpt_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")

            wandb_train(config_dict, model, ckpt_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim) # train the model

            #### perfomrm predictions for one needle ####
            num_sys = 1
            num_haystack_examples = 50

            print(f"\n\nstarting predictions for {parameter} = {value}")

            config.override("num_haystack_examples", num_haystack_examples)
            config.override("num_sys_haystack", num_sys)
            
            _, _, _ = set_config_params(config, model_name) #Not sure if this matters
            steps_in = list(range(1,11))

            if ortho_haar:
                config.override("val_dataset_typ", "ortho_haar")
            
            

            ckpt_pred_steps = gen_ckpt_pred_steps(model_name)

            colors=['#000000', '#005CAB', '#E31B23', '#FFC325', '#00A651', '#9B59B6']
            config.override("needle_in_haystack", True)
            config.override("needle_final_seg_extended", False)

            ys, sim_objs = get_test_data(config, output_dir, num_haystack_examples)

            print(f"output dir: {output_dir}")
            print(f"config.use_pos_emb: {config.use_pos_emb}")

            print(f"shape of ys before predict_all_checkpoints: {ys.shape}")

            kal_step = predict_all_checkpoints(config, ckpt_dir, output_dir, logscale, ys, sim_objs, model_name)

            print(f"plotting train_conv convergence plots for haystack len {num_sys}")
            pred_ckpt_step = haystack_plots_train_conv_full(config, model_name, num_sys, output_dir, ckpt_dir, experiment_name, ckpt_pred_steps, kal_step, steps_in, colors, compute_more=make_preds, abs_err=abs_err)

            # clean device for new run
            del model
            torch.cuda.empty_cache()
            '''



    else:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\ndevice:", device)
        torch.cuda.empty_cache()
        # instantiate gpt2 model
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                    n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb)
        
        model.to(device)
        
        output_dir, ckpt_dir, experiment_name = setup_train(model, train_mix_dist, train_mix_state_dim)
        # output_dir = output_dir + f"_{config.dataset_typ}{config.C_dist}"

        #update code for checking if training data exists
        # os.makedirs(output_dir + f"/data/", exist_ok=True)

        # if part_train_set:
        #     start_path_str = "../outputs/GPT2/"
        #     prev_exper = start_path_str + "241103_013426.749aca_gaussA_gauss_C_lr_0" #path to previous experiment to load

        #     #load training data
        #     with open(prev_exper + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl", "rb") as f:
        #         past_train_set = pickle.load(f)

        #     new_train_set = past_train_set[0:config.num_tasks]
        #     print("len of past train set:", len(past_train_set))
        #     print("len of new train set:", len(new_train_set))

        #     with open(output_dir + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
        #         pickle.dump(new_train_set, f)

        #     with open(prev_exper + f"/data/train_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "rb") as f:
        #         past_train_sim_objs = pickle.load(f)

        #     new_train_sim_objs = past_train_sim_objs[0:config.num_tasks]
        #     print("len of past train sim objs:", len(past_train_sim_objs))
        #     print("len of new train sim objs:", len(new_train_sim_objs))

        #     with open(output_dir + f"/data/train_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
        #         pickle.dump(new_train_sim_objs, f)

        #     del past_train_set
        #     del new_train_set
        #     del past_train_sim_objs
        #     del new_train_sim_objs
        #     torch.cuda.empty_cache()
        #     gc.collect()
               
        # else:
        #     collect_data(config, output_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, train_mix_C=train_mix_C) # collect data

        # replace ckpt_path with the path to the checkpoint file
        if config.acc:
            ckpt_dir = f"../../{experiment_name}"

        config.override("ckpt_path", ckpt_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")

        wandb_train(config, config_dict, model, ckpt_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim) # train the model

        # create prediction plots
        run_preds = True #run the predictions evaluation
        run_deg_kf_test = False #run degenerate KF test
        excess = False #run the excess plots
        shade = True


        if not config.acc:
            # get the sim objs for the validation data
            with open(output_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}_sim_objs.pkl", "rb") as f:
                sim_objs = pickle.load(f)

            #set ys to be the validation data
            with open(output_dir + f"/data/val_{config.val_dataset_typ}{config.C_dist}_state_dim_{config.nx}.pkl", "rb") as f:
                samples = pickle.load(f)
                # for every 2000 entries in samples, get the observation values and append them to the ys list
                ys = np.stack(
                    [entry["obs"][:config.n_positions + 1] for entry in samples], axis=0
                ).reshape((config.num_val_tasks, config.num_traces["val"], config.n_positions + 1, config.ny)).astype(np.float32)

                gc.collect()  # Start the garbage collector

            print("ckpt_path", config.ckpt_path)
            create_plots(config, model, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade, logscale=logscale, train_conv=train_conv, tf=tf, ys=ys, sim_objs=sim_objs, output_dir=output_dir)