# import numpy as np
import dyn_models
import pickle
import numpy

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
with open("../outputs/GPT2/241117_204332.cee615_upperTriA_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000/data/val_gaussA_gauss_C_state_dim_10_sim_objs.pkl", "rb") as f:
        sim_objs = pickle.load(f)

print("len of sim_objs", len(sim_objs))
sys_ind = 500
print(f"type(sim_objs[{str(sys_ind)}]])", type(sim_objs[sys_ind]))
#print the first system
print(f"sim_objs[{str(sys_ind)}].A", sim_objs[sys_ind].A)
#check if the first system is upper triangular and print
print("is upper triangular", numpy.allclose(sim_objs[sys_ind].A, numpy.triu(sim_objs[sys_ind].A)))
#print just the diagonal elements of the first system
print("diagonal elements", numpy.diagonal(sim_objs[sys_ind].A))
print("\n\n\n")


# #convert the systems to tensors
# tensor_dicts = convert_to_tensor_dicts(sim_objs)

# print("tensor_dicts", tensor_dicts)

# #save the tensors to a file in the same directory
# with open("../outputs/GPT2/240619_070456.1e49ad_upperTriA_gauss_C/data/val_upperTriA_gauss_C_sim_objs.pkl", "wb") as f:
#         pickle.dump(tensor_dicts, f)


