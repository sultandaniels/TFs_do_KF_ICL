import os
import shutil

def move_files_and_directories(directories, model_names, destination_base):
    for directory, model_name in zip(directories, model_names):
        destination_dir = os.path.join(destination_base, model_name)
        
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)
        
        # Move PDF files
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.pdf'):
                    src_file = os.path.join(dirpath, filename)
                    dest_file = os.path.join(destination_dir, filename)
                    try:
                        shutil.move(src_file, dest_file)
                        print(f"Moved: {src_file} to {dest_file}")
                    except Exception as e:
                        print(f"Error moving {src_file} to {dest_file}: {e}")
        
        # Move the 'train' subdirectory
        train_src = os.path.join(directory, 'train')
        train_dest = os.path.join(destination_dir, 'train')
        if os.path.exists(train_src):
            try:
                shutil.move(train_src, train_dest)
                print(f"Moved directory: {train_src} to {train_dest}")
            except Exception as e:
                print(f"Error moving directory {train_src} to {train_dest}: {e}")

def list_dirs(path):
    return sorted([os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])

# Example usage
directories = [
    '/path/to/dir1',
    '/path/to/dir2',
    '/path/to/dir3'
]

model_names = [
    'model1',
    'model2',
    'model3'
]

destination_base = '/Users/sultandaniels/Documents/Transformer_Kalman/67b648fddceb6a3b3812dd6c/Arxiv_Interleaved Time-series show the Emergence of Associative Recall is no Mirage/needle/train_conv/seg_len_10'

# move_files_and_directories(directories, model_names, destination_base)

# List all directories in a path
ls = list_dirs("/home/sultand/TFs_do_KF_ICL/outputs/GPT2")
print(ls)