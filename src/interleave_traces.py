from create_plots_with_zero_pred import interleave_traces
#import the config
from core import Config
from data_train import get_entries
import pickle

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

    ys = get_entries(config, path + f"/data/val_{valA}{valC}_state_dim_{nx}.pkl")

    #set num_sys_haystack
    num_sys_haystack = 1 #number of systems in the haystack
    config.override("num_sys_haystack", num_sys_haystack)
    #set num_test_traces_configs to num_sys_haystack
    config.override("num_test_traces_configs", num_sys_haystack)

    #get interleaved traces
    multi_sys_ys, sys_choices_per_config, sys_dict_per_config, tok_seg_lens_per_config, seg_starts_per_config, real_seg_lens_per_config, sys_inds_per_config = interleave_traces(config, ys, num_test_traces_configs=config.num_test_traces_configs, num_trials=1000)

    #save the outputs of interleave traces to a zipped file
    file_dict = {"multi_sys_ys": multi_sys_ys, "sys_choices_per_config": sys_choices_per_config, "sys_dict_per_config": sys_dict_per_config, "tok_seg_lens_per_config": tok_seg_lens_per_config, "seg_starts_per_config": seg_starts_per_config, "real_seg_lens_per_config": real_seg_lens_per_config, "sys_inds_per_config": sys_inds_per_config}

    filename = path + f"/data/interleaved_traces_{valA}{valC}_state_dim_{nx}_num_sys_haystack_{num_sys_haystack}.zip"
    with open(filename, 'wb') as f:
        pickle.dump(file_dict, f)
    print(f"Saved interleaved traces to {filename}")
    

