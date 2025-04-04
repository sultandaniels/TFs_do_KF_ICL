import os
import re
import shutil

def delete_files(directory, suffix):
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            # print(f"filename: {filename}")
            # if filename == suffix:

            if filename.startswith(suffix):
                filepath = os.path.join(directory, filename)
                os.remove(filepath)
                print(f"Deleted: {filepath}\n\n\n")
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
    # directory = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"  # Replace with the path to your directory
    # # suffix = f"50.ckpt"  # The suffix of the files you want to delete
    # prefix = f"train_conv_needle_haystack_len_"

    # # delete_files(directory, suffix)

    # # delete_files_spec_interval(directory)

    # for ckpt in range(3000, 180000, 3000):
    #     for haystack_len in range(2,3):
    #         delete_files(directory + f"/prediction_errors_ident_C_step={str(ckpt)}.ckpt/", f"train_conv_needle_haystack_len_{haystack_len}_val")
    
    # # delete_files(directory, suffix)


    # Define the base directory
    base_dir = "/home/sultand/TFs_do_KF_ICL/outputs/GPT2"

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            # Check if the directory is 'checkpoints' or 'data'
            if dir_name in ["checkpoints", "data"]:
                dir_path = os.path.join(root, dir_name)
                try:
                    # Delete the directory
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")
                except Exception as e:
                    print(f"Failed to delete {dir_path}: {e}")