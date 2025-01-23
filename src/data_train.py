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

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
os.environ["WANDB_SILENT"] = "true"

def wandb_train(config_dict, model, output_dir, train_mix_dist=False, train_mix_state_dim=False):

    # test_dataset_typ = config.dataset_typ
    # test_C_dist = config.C_dist
    # # add ckpt_path to config_dict
    # config_dict["ckpt_path"] = config.ckpt_path
    # config_dict["dataset_typ"] = "gaussA"
    # config_dict["C_dist"] = "_gauss_C"

    # #change dataset_typ and C_dist in config to "gaussA" and "_gauss_C" 
    # config.override("dataset_typ", "gaussA")
    # config.override("C_dist", "_gauss_C")

    # 🐝 1️⃣ Start a new run to track this script
    run = wandb.init(
        # Set the project where this run will be logged
        project="transformer_kalman_no_sweep",
        # Track hyperparameters and run metadata
        config=config_dict,
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
    )
    train_time = train_gpt2(model, config, output_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim) # train the model

    # Finish the run
    wandb.finish()

    # Get the path to the current run directory
    run_dir = run.dir

    # Delete the run directory
    shutil.rmtree(run_dir)

    print(f"Deleted wandb run directory: {run_dir}")

    # #change dataset_typ and C_dist back to the original values
    # config.override("dataset_typ", test_dataset_typ)
    # config.override("C_dist", test_C_dist)
    return None

def preds_thread(config, ckpt_path, make_preds, resume_train, train_conv, logscale, tf, train_mix_dist=False, train_mix_state_dim=False):
    # create prediction plots
    run_preds = make_preds # run the predictions evaluation
    run_deg_kf_test = False #run degenerate KF test
    excess = False #run the excess plots
    shade = True
    config.override("ckpt_path", ckpt_path)
    print("\n\n\n config.ckpt_path:", config.ckpt_path)
    print("testing config.nx:", config.nx)
    print("\n\n")

    if resume_train:
        #get the parent directory of the ckpt_path
        parent_dir = os.path.dirname(config.ckpt_path)
        #get the parent directory of the parent directory
        output_dir = os.path.dirname(parent_dir)
        # instantiate gpt2 model
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
        
        wandb_train(config_dict, model, output_dir, train_mix_dist, train_mix_state_dim)

    create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade, logscale=logscale, train_conv=train_conv, tf=tf)

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


def predict_all_checkpoints(config, output_dir, logscale):
        config.override("num_test_traces_configs", 1)
        config.override("needle_in_haystack", False)
        config.override("single_system", True)
        filecount = 0
        for filename in os.listdir(output_dir + "/checkpoints/"):
            filecount += 1
            print("filecount:", filecount)
            ckpt_path = output_dir + "/checkpoints/" + filename
            print("ckpt_path:", ckpt_path)
            run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds=True, resume_train=False, train_conv=True, logscale=logscale, tf=True)


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
    parser.add_argument('--kfnorm', help='Boolean. subtract kalman performance from error', action='store_true')
    parser.add_argument('--olsnorm', help='Boolean. subtract kalman performance from error', action='store_true')
    parser.add_argument('--t_conv_plot', help='Boolean. plot the convergence plots with t as the indep. var.', action='store_true')
    parser.add_argument('--logscale', help='Boolean. plot the system test evaluations on a logscale with KF err subtracted.', action='store_true')
    parser.add_argument('--logscale_an', help='Boolean. plot the system test evaluations on a logscale divided by analytical KF err.', action='store_true')
    parser.add_argument('--train_mix_dist', help='Boolean. generate training data from a mixture of gaussian, uppertriA, and rotdiagA', action='store_true')
    parser.add_argument('--train_mix_state_dim', help='Boolean. generate training data from a mixture of state dimensions', action='store_true')
    parser.add_argument('--train_mix_C', help='Boolean. generate training data from a mixture of C dists', action='store_true')
    parser.add_argument('--part_train_set', help='Boolean. train on a subset of a previous experiments train dataset', action='store_true')

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


    config = Config() # create a config object
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

    if (not train_conv) and (make_preds or saved_preds or resume_train):
        ckpt_path = "../outputs/GPT2/250114_202420.3c1184_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000/checkpoints/step=141000.ckpt"
        
        run_preds, run_deg_kf_test, excess, shade = preds_thread(config, ckpt_path, make_preds, resume_train, train_conv, logscale, tf, train_mix_dist, train_mix_state_dim)
        
    elif train_conv:
        output_dir = "../outputs/GPT2/250114_202420.3c1184_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000"
        #241117_204226.922f5f_rotDiagA_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000" #241117_204332.cee615_upperTriA_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000"
        if make_preds:
            predict_all_checkpoints(config, output_dir, logscale)
        
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("\ndevice:", device)
        # instantiate gpt2 model
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                    n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb)
        
        model.to(device)
        
        output_dir = setup_train(model, train_mix_dist, train_mix_state_dim)
        # output_dir = output_dir + f"_{config.dataset_typ}{config.C_dist}"
        os.makedirs(output_dir + f"/data/", exist_ok=True)

        if part_train_set:
            start_path_str = "../outputs/GPT2/"
            prev_exper = start_path_str + "241103_013426.749aca_gaussA_gauss_C_lr_0" #path to previous experiment to load

            #load training data
            with open(prev_exper + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl", "rb") as f:
                past_train_set = pickle.load(f)

            new_train_set = past_train_set[0:config.num_tasks]
            print("len of past train set:", len(past_train_set))
            print("len of new train set:", len(new_train_set))

            with open(output_dir + f"/data/train_{config.dataset_typ}{config.C_dist}.pkl", "wb") as f:
                pickle.dump(new_train_set, f)

            with open(prev_exper + f"/data/train_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "rb") as f:
                past_train_sim_objs = pickle.load(f)

            new_train_sim_objs = past_train_sim_objs[0:config.num_tasks]
            print("len of past train sim objs:", len(past_train_sim_objs))
            print("len of new train sim objs:", len(new_train_sim_objs))

            with open(output_dir + f"/data/train_{config.dataset_typ}{config.C_dist}_sim_objs.pkl", "wb") as f:
                pickle.dump(new_train_sim_objs, f)

            del past_train_set
            del new_train_set
            del past_train_sim_objs
            del new_train_sim_objs
            torch.cuda.empty_cache()
            gc.collect()
               
        else:
            collect_data(config, output_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim, train_mix_C=train_mix_C) # collect data

        # replace ckpt_path with the path to the checkpoint file
        config.override("ckpt_path", output_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")

        wandb_train(config_dict, model, output_dir, train_mix_dist=train_mix_dist, train_mix_state_dim=train_mix_state_dim) # train the model

        # create prediction plots
        run_preds = True #run the predictions evaluation
        run_deg_kf_test = False #run degenerate KF test
        excess = False #run the excess plots
        shade = True

        print("ckpt_path", config.ckpt_path)
        create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade, logscale=logscale, train_conv=train_conv, tf=tf)