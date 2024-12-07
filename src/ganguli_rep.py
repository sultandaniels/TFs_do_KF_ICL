from collect_data import collect_data
from models import GPT2
from core import Config
from train import train_gpt2
from core import setup_train
import os
from create_plots_with_zero_pred import create_plots, save_preds
import argparse
# Log in to your W&B account
import wandb

# main function

if __name__ == '__main__':

    wandb.login()
    # # Create the parser
    # parser = argparse.ArgumentParser(description='Run Predictions or not.')

    # # Add the arguments
    # parser.add_argument('--saved_preds', help='Boolean. Just plot the errors for a previously evaluated checkpoint', action='store_true')

    # # Parse the arguments
    # args = parser.parse_args()

    # # Now you can use the flag
    # print(args.saved_preds)
    # saved_preds = args.saved_preds

    #create a list named Ms that starts at 2 and has 20 successive elements that are 2 times the previous element
    Ms = [2] 
    for i in range(19):
        Ms.append(Ms[-1]*2)
    print("list of num_tasks", Ms)
    for num_tasks in Ms: #iterate through the list of num_tasks
        config = Config() # create a config object
        config.override("num_tasks", num_tasks) #override the num_tasks attribute with the current num_tasks

        # instantiate gpt2 model
        model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                    n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
        
        output_dir = setup_train(model)
        # output_dir = output_dir + f"_{config.dataset_typ}{config.C_dist}"
        os.makedirs(output_dir + f"/data/", exist_ok=True)

        collect_data(model, config, output_dir) # collect data

        # replace ckpt_path with the path to the checkpoint file
        config.override("ckpt_path", output_dir + "/checkpoints/step=" + str(config.train_steps) + ".ckpt")
        
        # 🐝 1️⃣ Start a new run to track this script
        run = wandb.init(
            # Set the project where this run will be logged
            project="my-awesome-project",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.01,
                "epochs": 10,
            },
        )

        train_gpt2(model, config, output_dir) # train the model

        # create prediction plots
        run_preds = True #run the predictions evaluation
        run_deg_kf_test = False #run degenerate KF test
        excess = False #run the excess plots
        shade = True

        print("ckpt_path", config.ckpt_path)
        save_preds(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade)

        # create_plots(config, run_preds, run_deg_kf_test, excess, num_systems=config.num_val_tasks, shade=shade)