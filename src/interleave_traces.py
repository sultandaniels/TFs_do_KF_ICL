##############################
#multiline comment
""""
multi_sys_ys: np array, shape (num_test_traces_configs, num_traces_per_config, trace length, ny + special token dim), this holds the prompts
sys_choices_per_config: list of lists, shape (num_test_traces_configs, # of segments in the prompt), this holds the order of the systems in the haystack. The system indices correspond to the unintereaved system corpus
sys_dict_per_config: list of dictionaries, length num_test_traces_configs, each dictionary's keys are the system index in the entire uninterleaved system corpus and the values are the system index for the subset of systems chosen for each trace
tok_seg_lens_per_config: list of lists, shape (num_test_traces_configs, # of segments in the prompt), this holds the lengths of the segments in the interleaved traces including special tokens
seg_starts_per_config: list of lists, shape (num_test_traces_configs, # of segments in the prompt), this holds the starting indices of the segments in the interleaved traces
real_seg_lens_per_config: list of lists, shape (num_test_traces_configs, # of segments in the prompt), this holds the lengths of the segments in the interleaved traces excluding special tokens
sys_inds_per_config: list of lists, shape (num_test_traces_configs, # of systems chosen for trace), this holds the subset of systems chosen for each trace
"""

#####################



from create_plots_with_zero_pred import interleave_traces
#import the config
from core import Config
from data_train import get_entries
import pickle
import os

if __name__ == "__main__":
    config = Config()

    #get val data from "../outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"
    path = "../outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"

    valA = "ortho" #"ident", "ortho", "gaussA" #system family for linear systems

    if valA == "ortho" or valA == "ident":
        valC = "_ident_C"
        nx = 5 #state dimension
    elif valA == "gaussA":
        valC = "_gauss_C"
        nx = 10 #state dimension

    #get the data
    config.override("val_dataset_typ", valA)
    config.override("C_dist", valC)
    config.override("nx", nx)
    config.override("late_start", None) #index of the first time step to start interleaving with. If None, will start from the beginning of the trace
    config.override("num_val_tasks", 100) #number of validation systems
    config.override("num_traces", {"train": 1, "val": 1000}) #number of validation traces per system

    config.override("multi_sys_trace", True) #use multi system traces
    config.override("needle_in_haystack", True) #make needle in haystack prompts
    #set num_sys_haystack



    num_sys_haystack = 1 #number of systems in the haystack



    config.override("num_sys_haystack", num_sys_haystack)
    #set num_test_traces_configs to num_sys_haystack
    config.override("num_test_traces_configs", num_sys_haystack)
    config.override("n_positions", (config.len_seg_haystack + 2)*(num_sys_haystack+1)) #number of positions in the needle in haystack prompt

    val_path = path + f"/data/val_{valA}{valC}_state_dim_{nx}.pkl"

    print(f"Getting val data from {val_path}")
    with open(val_path, 'rb') as f:
        ys = get_entries(config, f) #get the uninterleaved validation data

    
    #get interleaved traces
    multi_sys_ys, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, real_seg_lens_per_config, sys_inds_per_config = interleave_traces(config, ys, num_test_traces_configs=config.num_test_traces_configs, num_trials=config.num_traces["val"], ex=0)

    #save the outputs of interleave traces to a zipped file
    file_dict = {"multi_sys_ys": multi_sys_ys, "sys_choices_per_config": sys_choices_per_config, "sys_dict_per_config": sys_dict_per_config, "tok_seg_lens_per_config": tok_seg_lens_per_config, "seg_starts_per_config": seg_starts_per_config, "real_seg_lens_per_config": real_seg_lens_per_config, "sys_inds_per_config": sys_inds_per_config}


    filename = path + f"/data/interleaved_traces_{valA}{valC}_state_dim_{nx}_num_sys_haystack_{num_sys_haystack}.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(file_dict, f)
    print(f"Saved interleaved traces to {filename}")
    

