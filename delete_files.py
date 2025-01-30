import os

def delete_files(directory, suffix):
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            # print(f"filename: {filename}")
            if filename == suffix:
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                print(f"Deleted: {filepath}")
    else:
        print(f"Directory does not exist: {directory}")
    return None

if __name__ == "__main__":
    directory = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/"  # Replace with the path to your directory
    suffix = f"50.ckpt"  # The suffix of the files you want to delete

    for ckpt in range(3000, 183000, 3000):
        for i in range(1, 100):
            for haystack_len in [1,2,3,4,9,14,19]:
                delete_files(directory + f"/prediction_errors_ident_C_step={str(ckpt)}.ckpt/", f"train_conv_needle_haystack_len_{haystack_len}_val_ortho_state_dim_5_sys_choices_sys_dict_tok_seg_lens_seg_starts_example_{i}.pkl")
    
    # delete_files(directory, suffix)