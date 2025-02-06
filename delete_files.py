import os
import re

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


def delete_files_spec_interval(directory):
    # Regular expression to match filenames like 'step=100.ckpt'
    pattern = re.compile(r'step=(\d+)\.ckpt')

    # List all files in the directory
    files = os.listdir(directory)

    # Separate files into two groups based on the step value
    files_to_keep = set()
    files_to_delete = set()

    for filename in files:
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            if step <= 5000:
                if step % 100 == 0:
                    files_to_keep.add(filename)
                else:
                    files_to_delete.add(filename)
            else:
                if step % 400 == 0:
                    files_to_keep.add(filename)
                else:
                    files_to_delete.add(filename)

    # print(f"Files to keep: {files_to_keep}")
    # print(f"\n\n\nFiles to delete: {files_to_delete}")
    # Delete the files that are not in the keep list
    for filename in files_to_delete:
        file_path = os.path.join(directory, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

if __name__ == "__main__":
    directory = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250124_052617.8dd0f8_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/checkpoints"  # Replace with the path to your directory
    suffix = f"50.ckpt"  # The suffix of the files you want to delete

    # delete_files(directory, suffix)

    delete_files_spec_interval(directory)

    # for ckpt in range(3000, 183000, 3000):
    #     for i in range(1, 100):
    #         for haystack_len in [1,2,3,4,9,14,19]:
    #             delete_files(directory + f"/prediction_errors_ident_C_step={str(ckpt)}.ckpt/", f"train_conv_needle_haystack_len_{haystack_len}_val_ortho_state_dim_5_sys_choices_sys_dict_tok_seg_lens_seg_starts_example_{i}.pkl")
    
    # delete_files(directory, suffix)