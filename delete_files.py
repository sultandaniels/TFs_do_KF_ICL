import os

def delete_files(directory, suffix):
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            # print(f"filename: {filename}")
            # if filename == suffix:

            if filename.endswith(suffix):
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                print(f"Deleted: {filepath}")
    else:
        print(f"Directory does not exist: {directory}")
    return None

if __name__ == "__main__":
    directory = "/home/sultand/lambda_labs_filesystem/GPT2/250125_202437.caf35b_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.3207437987531975e-05_num_train_sys_40000/checkpoints"  # Replace with the path to your directory
    suffix = f"50.ckpt"  # The suffix of the files you want to delete

    delete_files(directory, suffix)

    # for ckpt in range(3000, 183000, 3000):
    #     for i in range(1, 100):
    #         for haystack_len in [1,2,3,4,9,14,19]:
    #             delete_files(directory + f"/prediction_errors_ident_C_step={str(ckpt)}.ckpt/", f"train_conv_needle_haystack_len_{haystack_len}_val_ortho_state_dim_5_sys_choices_sys_dict_tok_seg_lens_seg_starts_example_{i}.pkl")
    
    # delete_files(directory, suffix)