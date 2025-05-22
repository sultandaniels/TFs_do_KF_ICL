import logging
from dyn_models import generate_lti_sample
from core import Config
from tqdm import tqdm
import pickle
import os
import numpy as np
from models import GPT2
from dyn_models.ortho_sync_data import gen_ortho_sync_data
import argparse

def mix_ind(i, total_number, labels, counts, max_ind=2):
    #currently function only works for 
    if int(np.floor(len(labels)*i/total_number)) > max_ind:
        index = len(labels) - 1
        print("greater than 2")
    else:
        index = int(np.floor(len(labels)*i/total_number))

    counts[index] += 1

    return index, counts

#modify collect data so that it can tolerate multiple traces for one system
def collect_data(config, output_dir, only="", train_mix_dist=False, train_mix_state_dim=False, train_mix_C=False, specific_sim_objs=None, opposite_ortho=False):

    
    if specific_sim_objs:
        only = "val"
        config.override("num_val_tasks", len(specific_sim_objs))
        config.override("val_dataset_typ", "specA")
        config.override("C_dist", "_spec_C")


    if opposite_ortho:
        config.override("val_dataset_typ", "ortho")
        config.override("num_val_tasks", 2)
        config.override("C_dist", "_ident_C")


    for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
        if only and name != only: #if only is specified, skip the other dataset
            continue
        samples = [] #make sure that train and val samples are different
        sim_objs = [] #make sure that train and val sim_objs are different
        print("Generating", num_tasks*config.num_traces[name], "samples for", name)

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
            
            if train_mix_C:
                C_dists = ["_gauss_C", "_zero_C"]
                C_counts = np.zeros(len(C_dists))

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

        if (name == "train" and config.dataset_typ == "ortho_sync") or (name == "val" and config.val_dataset_typ == "ortho_sync"): #ortho sync data

            context = config.n_positions + 1
            sync_ind = 10 #index in traces where the vector is synchronized

            samples, ortho_mats, sim_objs = gen_ortho_sync_data(num_tasks, config.nx, context, sync_ind, config.num_traces[name])

        else:
        
            for i in tqdm(range(num_tasks)):
                if name == "train": 
                    if train_mix_dist:
                        dist_index, dist_counts = mix_ind(i, num_tasks, A_dists, dist_counts)

                        config.override("dataset_typ", A_dists[dist_index]) #override the dataset_typ
                    
                    if train_mix_state_dim:
                        if train_mix_dist:
                            total_tasks = num_tasks/np.ceil(len(A_dists))
                            ind = i
                            while ind > total_tasks:
                                ind -= total_tasks
                        else:
                            total_tasks = num_tasks
                            ind = i
                        
                        state_index, state_counts = mix_ind(ind, total_tasks, nxs, state_counts)
                        config.override("nx", nxs[state_index]) #override the nx
                    
                    if train_mix_C:
                        C_index, C_counts = mix_ind(i, num_tasks, C_dists, C_counts)

                        config.override("C_dist", C_dists[C_index]) #override the dataset_typ

                if opposite_ortho and i > 0:
                    #get the A from sim_objs and negate it
                    fsim = sim_objs[0]
                    fsim.A = -fsim.A

                    specific_sim_objs = [fsim]
                    
                fsim, sample = generate_lti_sample(config.C_dist, config.dataset_typ if name == "train" else config.val_dataset_typ, config.num_traces[name], config.n_positions, config.nx, config.ny, sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise, cond_num=cond_nums[int(np.floor(config.distinct_cond_nums*i/num_tasks))] if ((name == "train" and config.dataset_typ == "cond_num") or (name == "val" and config.val_dataset_typ == "cond_num")) else None, specific_sim_obj=(specific_sim_objs[0] if opposite_ortho and specific_sim_objs else (specific_sim_objs[i] if specific_sim_objs else None)))

                if (name == "train" and config.dataset_typ == "cond_num") or (name == "val" and config.val_dataset_typ == "cond_num"):
                    cond_counts[int(np.floor(config.distinct_cond_nums*i/num_tasks))] += 1

                #the samples are partioned by the number of traces for each system.
                
                samples.extend([{k: v[i] for k, v in sample.items()} for i in range(config.num_traces[name])])
                # raise Exception("just checking fsim type umich_meta_output_predictor/src/collect_data.py")
                if opposite_ortho:
                    print(f"fsim.A: {fsim.A}")
                sim_objs.append(fsim)

        print("Saving", len(samples), "samples for", name)

        loc = f"{output_dir}/"  + ("for_multi_cut_" if config.multi_cut_val else "") + ("opposite_ortho_" if opposite_ortho else "") + ("train_systems_" if specific_sim_objs and not opposite_ortho else "") +f"{name}_" + (f"{config.dataset_typ}" if name == "train" else f"{config.val_dataset_typ}") + f"{config.C_dist}" + f"_state_dim_{config.nx}" + ("_dist_mix" if train_mix_dist and name == "train" else "") + ("_state_dim_mix" if train_mix_state_dim and name == "train" else "") + (f"_sync_ind_{sync_ind}" if name == "train" and config.dataset_typ == "ortho_sync" or name == "val" and config.val_dataset_typ == "ortho_sync" else "")

        with open(loc + ".pkl", "wb") as f:
            pickle.dump(samples, f)

        print("location:", loc + ".pkl")

        print("output_dir:", output_dir)
        #save fsim to pickle file
        with open(loc + "_sim_objs.pkl", "wb") as f:
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
        if train_mix_dist:
            mode = 'a'
        else:
            mode = 'w'
        with open(output_dir + "/train_dist_mix_record.txt", mode) as file:
        # Write the print statement to the file
            for k in range(len(nxs)):
                file.write(f"state dim {str(nxs[k])} count: {state_counts[k]}\n")
            file.close()

    if train_mix_C:
        if train_mix_dist or train_mix_state_dim:
            mode = 'a'
        else:
            mode = 'w'
        with open(output_dir + "/train_dist_mix_record.txt", mode) as file:
        # Write the print statement to the file
            for k in range(len(C_dists)):
                file.write(f"state dim {str(C_dists[k])} count: {C_counts[k]}\n")
            file.close()



if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='What data to collect.')

    # Add the arguments
    parser.add_argument('--val', help='Boolean. only generate validation data', action='store_true')
    parser.add_argument('--train', help='Boolean. only generate training data', action='store_true')
    parser.add_argument('--train_mix_dist', help='Boolean. generate training data from gaussian, uppertriA, and rotdiagA', action='store_true')
    parser.add_argument('--train_mix_state_dim', help='Boolean. generate training data from a mixture of state dimensions', action='store_true')
    parser.add_argument('--opposite_ortho', help='Boolean. generate training data from opposite orthogonal systems', action='store_true')


    # Parse the arguments
    args = parser.parse_args()
    print("only val:", args.val)
    print("only train:", args.train)
    print("train_mix_dist:", args.train_mix_dist)
    print("train_mix_state_dim:", args.train_mix_state_dim)
    print("opposite_ortho:", args.opposite_ortho)

    train_mix_dist = args.train_mix_dist
    train_mix_state_dim = args.train_mix_state_dim
    opposite_ortho = args.opposite_ortho

    # Now you can use the flag
    if args.val:
        only = "val"
    elif args.train:
        only = "train"
    else:
        only = ""

    config = Config()
    
    collect_data(config, "/data/shared/ICL_Kalman_Experiments/train_and_test_data/ortho_sync", only, train_mix_dist, opposite_ortho=opposite_ortho)