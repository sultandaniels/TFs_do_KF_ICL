import os
from src.data_processing import gen_ckpt_steps

import re

def rename_late_start_files(directory):
    pattern = re.compile(r'late_start_(\d+)([a-zA-Z])')
    
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            new_filename = pattern.sub(r'late_start_\1_\2', filename)
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {old_filepath} -> {new_filepath}")


def rename_files(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            # find the index in the filename where the prefix ends
            # print(f"prefix: {prefix}\n filename: {filename}\n new filename: {filename[:len(prefix)] + filename[len(prefix)+11:]}\n\n\n")
            # print(f"prefix: {prefix}\n filename: {filename}\n new filename: {filename[:18] + "haystack_len_4_" + filename[18:42] + filename[53:]}\n\n\n")
            new_filename = filename[:18] + "haystack_len_4_" + filename[18:42] + filename[53:]
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {old_filepath} -> \n{new_filepath}\n\n\n\n")
            # print(f"Renamed: {filename} -> {new_filename}\n\n\n\n")

if __name__ == "__main__":

    # ckpts = gen_ckpt_steps(108000, 180000, 3000)
    # for ckpt in ckpts:
    #     directory = f"/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250114_202420.3c1184_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000/prediction_errors_gauss_C_step={ckpt}.ckpt"  # Replace with the path to your directory
    #     prefix = "train_conv_needle_val_gaussA_state_dim_10_"  # Replace with the old prefix

    #     if os.path.exists(directory):
    #         rename_files(directory, prefix)
    #     else:
    #         print(f"path: {directory} does not exist.")

        
    # directory = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/needles/train_conv"  # Replace with the path to your directory

    # if os.path.exists(directory):
    #     rename_late_start_files(directory)
    # else:
    #     print(f"Directory {directory} does not exist.")

    GPT2_dir = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2/"
    for filename in os.listdir(GPT2_dir):
        directory = GPT2_dir + filename + "/needles/train_conv"  # Replace with the path to your directory

        if os.path.exists(directory):
            rename_late_start_files(directory)
        else:
            print(f"Directory {directory} does not exist.")
    