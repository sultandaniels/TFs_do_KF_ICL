import os
from src.data_processing import gen_ckpt_steps

def rename_files(directory, prefix):
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            #find the index in the filename where the prefix ends
            # print(f"prefix: {prefix}\n filename: {filename}\n new filename: {filename[:len(prefix)] + filename[len(prefix)+11:]}\n\n\n")
            # print(f"prefix: {prefix}\n filename: {filename}\n new filename: {"train_conv_" + filename[:7] + "haystack_len_4_" + filename[7:]}\n\n\n")
            new_filename = "train_conv_" + filename[:7] + "haystack_len_4_" + filename[7:]
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {old_filepath} -> \n{new_filepath}\n\n\n\n")

if __name__ == "__main__":

    ckpts = gen_ckpt_steps(3000, 105000, 3000)
    for ckpt in ckpts:
        directory = f"/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/prediction_errors_ident_C_step={ckpt}.ckpt"  # Replace with the path to your directory
        prefix = "needle_val_ortho_state_dim_5_"  # Replace with the old prefix

        if os.path.exists(directory):
            rename_files(directory, prefix)
        else:
            print(f"path: {directory} does not exist.")