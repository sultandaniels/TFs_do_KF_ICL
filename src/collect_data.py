import logging
from dyn_models import generate_lti_sample
from core import Config
from tqdm import tqdm
import pickle
import os
import numpy as np
from models import GPT2
import argparse

def mix_ind(i, total_number, labels, counts):
    #currently function only works for 
    if int(np.floor(len(labels)*i/total_number)) > 2:
        index = 0
        print("greater than 2")
    else:
        index = int(np.floor(len(labels)*i/total_number))

    counts[index] += 1

    return index, counts

#modify collect data so that it can tolerate multiple traces for one system
def collect_data(model, config, output_dir, only="", train_mix_dist=False, train_mix_state_dim=False):

    logger = logging.getLogger(__name__)
    # config = Config()


    # instantiate gpt2 model (FOR THE SAKE OF TESTING, REMOVE LATER)
    # model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
    #              n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)

    zero_count = 0
    one_count = 0
    two_count = 0
    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        if only and name != only: #if only is specified, skip the other dataset
            continue
        samples = [] #make sure that train and val samples are different
        sim_objs = [] #make sure that train and val sim_objs are different
        print("Generating", num_tasks, "samples for", name)

        if name == "train":
            
            if train_mix_dist:
                #distributions in train mix
                A_dists = ["gaussA", "upperTriA"] #"rotDiagA"] 
                print("Collecting training data from", A_dists, config.C_dist)
                dist_counts = np.zeros(len(A_dists))

            if train_mix_state_dim:
                #different state dims
                nxs = [3,10,20]
                state_counts = np.zeros(len(nxs))

        elif name == "train" and not (train_mix_dist or train_mix_state_dim):
            print("Collecting training data from", config.dataset_typ, config.C_dist)
        elif name == "val":
            print("Collecting validation data from", config.val_dataset_typ, config.C_dist)

        if ((name == "train" and config.dataset_typ == "cond_num") or (name == "val" and config.val_dataset_typ == "cond_num")):
            #set a list with 10 integer values from 1 to max_cond_num
            cond_nums = np.linspace(0, config.max_cond_num, config.distinct_cond_nums + 1, dtype=int)
            cond_nums = cond_nums[1:] #remove the first element which is 0
            print("cond_num:", cond_nums)
            #setup counters for each distinct cond_num
            cond_counts = np.zeros(config.distinct_cond_nums)
      
        for i in tqdm(range(num_tasks)):
            if name == "train": 
                if train_mix_dist:
                    dist_index, dist_counts = mix_ind(i, num_tasks, A_dists, dist_counts)

                    config.override("dataset_typ", A_dists[dist_index]) #override the dataset_typ
                
                if train_mix_state_dim:
                    state_index, state_counts = mix_ind(i, num_tasks, nxs, state_counts)
                    config.override("nx", nxs[state_index]) #override the nx
                
            fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ if name == "train" else config.val_dataset_typ, config.num_traces[name], config.n_positions, config.nx, config.ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise, cond_num=cond_nums[int(np.floor(config.distinct_cond_nums*i/num_tasks))] if ((name == "train" and config.dataset_typ == "cond_num") or (name == "val" and config.val_dataset_typ == "cond_num")) else None)

            if (name == "train" and config.dataset_typ == "cond_num") or (name == "val" and config.val_dataset_typ == "cond_num"):
                cond_counts[int(np.floor(config.distinct_cond_nums*i/num_tasks))] += 1
                    
            # repeated_A = np.repeat(sample["A"][np.newaxis,:,:], config.num_traces[name], axis=0) #repeat the A matrix for each trace
            # sample["A"] = repeated_A #repeat the A matrix for each trace

            # repeated_C = np.repeat(sample["C"][np.newaxis,:,:], config.num_traces[name], axis=0) #repeat the C matrix for each trace
            # sample["C"] = repeated_C #repeat the C matrix for each trace

            #the samples are partioned by the number of traces for each system.
            
            samples.extend([{k: v[i] for k, v in sample.items()} for i in range(config.num_traces[name])])
            # raise Exception("just checking fsim type umich_meta_output_predictor/src/collect_data.py")
            sim_objs.append(fsim)
        print("Saving", len(samples), "samples for", name)

        with open(output_dir + f"/data/{name}_" + (f"{config.dataset_typ}" if name == "train" else f"{config.val_dataset_typ}") + f"{config.C_dist}" + f"_state_dim_{config.nx}" + ("_dist_mix" if train_mix_dist and name == "train" else "") + ("_state_dim_mix" if train_mix_state_dim and name == "train" else "")+ ".pkl", "wb") as f:
            pickle.dump(samples, f)

        print("location:", output_dir + f"/data/{name}_" + (f"{config.dataset_typ}" if name == "train" else f"{config.val_dataset_typ}") + f"{config.C_dist}" + f"_state_dim_{config.nx}" + ("_mix" if train_mix_dist and name == "train" else "") + ("_state_dim_mix" if train_mix_state_dim and name == "train" else "") + ".pkl")

        print("output_dir:", output_dir)
        #save fsim to pickle file
        with open(output_dir + f"/data/{name}_" + (f"{config.dataset_typ}" if name == "train" else f"{config.val_dataset_typ}") + f"{config.C_dist}" + f"_state_dim_{config.nx}" + ("_mix" if train_mix_dist and name == "train" else "") + ("_state_dim_mix" if train_mix_state_dim and name == "train" else "") + "_sim_objs.pkl", "wb") as f:
            pickle.dump(sim_objs, f)

        if (config.dataset_typ == "cond_num" and name == "train") or (config.val_dataset_typ == "cond_num" and name == "val"):
            for i in range(config.distinct_cond_nums):
                print("cond_num:", cond_nums[i], "count:", cond_counts[i])

    if train_mix_dist:
        with open(output_dir + "/train_dist_mix_record.txt", 'w') as file:
        # Write the print statement to the file
            for k in range(len(A_dists)):
                file.write(f"{A_dists[k]} count: {dist_counts[k]}\n")
            file.close()

    if train_mix_state_dim:
        with open(output_dir + "/train_dist_mix_record.txt", 'w') as file:
        # Write the print statement to the file
            for k in range(len(nxs)):
                file.write(f"state dim {str(nxs[k])} count: {state_counts[k]}\n")
            file.close()

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='What data to collect.')

    # Add the arguments
    parser.add_argument('--val', help='Boolean. only generate validation data', action='store_true')
    parser.add_argument('--train', help='Boolean. only generate training data', action='store_true')
    parser.add_argument('--train_mix_dist', help='Boolean. generate training data from gaussian, uppertriA, and rotdiagA', action='store_true')
    parser.add_argument('--train_mix_state_dim', help='Boolean. generate training data from a mixture of state dimensions', action='store_true')


    # Parse the arguments
    args = parser.parse_args()
    print("only val:", args.val)
    print("only train:", args.train)
    print("train_mix_dist:", args.train_mix_dist)
    print("train_mix_state_dim:", args.train_mix_state_dim)

    train_mix_dist = args.train_mix_dist
    train_mix_state_dim = args.train_mix_state_dim

    # Now you can use the flag
    if args.val:
        only = "val"
    elif args.train:
        only = "train"
    else:
        only = ""

    config = Config()
    model = GPT2(config.n_dims_in, config.n_positions, n_dims_out=config.n_dims_out,
                 n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head)
    
    collect_data(model, config, "../outputs/GPT2/241101_190253.47c70e_gaussA_gauss_C_lr_6e-05", only, train_mix_dist)