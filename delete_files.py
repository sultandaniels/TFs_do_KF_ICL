import os

def delete_files(directory, suffix):
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)
            print(f"Deleted: {filepath}")

if __name__ == "__main__":
    directory = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250125_202437.caf35b_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.3207437987531975e-05_num_train_sys_40000/checkpoints"  # Replace with the path to your directory
    suffix = "50.ckpt"  # The suffix of the files you want to delete
    
    delete_files(directory, suffix)