# import numpy as np
import dyn_models
import pickle
import numpy
import linalg_helpers as la

import torch


def convert_to_tensor_dicts(sim_objs):
        tensor_dicts = []  # Initialize an empty list for dictionaries
        for sim_obj in sim_objs:
                # Convert .A and .C to tensors and create a dictionary
                tensor_dict = {
                        'A': torch.from_numpy(sim_obj.A),
                        'C': torch.from_numpy(sim_obj.C)
                }
                tensor_dicts.append(tensor_dict)  # Append the dictionary to the list
        return tensor_dicts



#check systems that were trained and validated
with open("../outputs/GPT2/250114_202420.3c1184_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000/data/val_gaussA_gauss_C_state_dim_10.pkl", "rb") as f:
        samples = pickle.load(f)

print(f"Number of samples: {len(samples)}")
print(f"Num of traces per sample: {len(samples)/25}")
la.print_matrix(samples[2000*18+1000]['obs'], "trace")


