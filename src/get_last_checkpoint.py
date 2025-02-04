import os
import re

def get_last_checkpoint(directory):
    # Regular expression to match filenames like 'step=1000.ckpt'
    pattern = re.compile(r'step=(\d+)\.ckpt')

    # List comprehension to find all matching files and extract their step values
    files = [(int(pattern.match(f).group(1)), f) for f in os.listdir(directory) if pattern.match(f)]

    # Find the file with the maximum step value
    if files:
        max_step_file = max(files, key=lambda x: x[0])[1]
        return max_step_file
    else:
        return None

def split_path(path):
    # Split the path into its components
    parts = path.split('/')

    # Join the first three parts with '/' and the rest separately
    first_part = '/'.join(parts[:3]) + "/"
    second_part = '/'.join(parts[3:])

    return first_part, second_part


if __name__ == "__main__":
    # Example usage
    directory = '/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250124_052617.8dd0f8_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/checkpoints'
    last_checkpoint = get_last_checkpoint(directory)
    if last_checkpoint:
        print(f"The last checkpoint file is: {last_checkpoint}")
    else:
        print("No checkpoint files found.")