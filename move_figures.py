import os
import shutil

def move_files_and_directories(directories, model_names, destination_base):
    print(f"Moving files and directories to {destination_base}")
    for directory, model_name in zip(directories, model_names):
        print(f"Directory: {directory}")
        destination_dir = os.path.join(destination_base, model_name)
        print(f"Destination directory: {destination_dir}\n\n")
        
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)
        
        # Move PDF files
        for dirpath, dirnames, filenames in os.walk(directory):
            print(f"Current directory: {dirpath}")
            print(f"Subdirectories: {dirnames}")
            print(f"Files: {filenames}")
            for filename in filenames:
                print(f"Filename: {filename}")
                if filename.endswith('.pdf'):
                    print(f"Found PDF: {os.path.join(dirpath, filename)}")
                    src_file = os.path.join(dirpath, filename)
                    #check if src_file has "/train/" in it
                    if "/train/" in src_file:
                        dest_file = os.path.join(destination_dir, 'train', filename)
                    else:
                        dest_file = os.path.join(destination_dir, filename)
                    try:
                        shutil.move(src_file, dest_file)
                        print(f"Moved test: {src_file} to {dest_file}\n\n\n")
                    except Exception as e:
                        print(f"Error moving test {src_file} to {dest_file}: {e}")
        
        # # Move the 'train' subdirectory
        # train_src = os.path.join(directory, 'train')
        # train_dest = os.path.join(destination_dir, 'train')
        # if os.path.exists(train_src):
        #     print(f"Found train directory: {train_src}")
        #     try:
        #         # shutil.move(train_src, train_dest)
        #         print(f"Moved train directory: {train_src} to {train_dest}")
        #     except Exception as e:
        #         print(f"Error moving train directory {train_src} to {train_dest}: {e}")

# Example usage
directories = ['../TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250114_202420.3c1184_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.584893192461114e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250124_052617.8dd0f8_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250125_103302.919337_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_3.169786384922228e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250125_104123.f75c04_multi_sys_trace_ortho_state_dim_5_ident_C_lr_3.169786384922228e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250125_110549.80eba5_multi_sys_trace_ident_state_dim_5_ident_C_lr_3.169786384922228e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250125_202437.caf35b_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_1.3207437987531975e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250125_204545.a2cee4_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250125_210849.09203d_multi_sys_trace_ident_state_dim_5_ident_C_lr_1.3207437987531975e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250128_022150.04b6bf_multi_sys_trace_gaussA_state_dim_10_gauss_C_lr_6.339572769844456e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250128_022310.fc649a_multi_sys_trace_ident_state_dim_5_ident_C_lr_6.339572769844456e-05_num_train_sys_40000', '../TFs_do_KF_ICL/outputs/GPT2/250128_022331.067361_multi_sys_trace_ortho_state_dim_5_ident_C_lr_6.339572769844456e-05_num_train_sys_40000']

print(len(directories))

model_names = [
    'ortho_med',
    'gauss_med',
    'ident_med',
    'gauss_small',
    'ortho_small',
    'ident_small',
    'gauss_big',
    'ortho_big',
    'ident_big',
    'gauss_tiny',
    'ident_tiny',
    'ortho_tiny',
]



destination_base = '/Users/sultandaniels/Documents/Transformer_Kalman/67b648fddceb6a3b3812dd6c/Arxiv_Interleaved Time-series show the Emergence of Associative Recall is no Mirage/needle/train_conv/seg_len_10'

move_files_and_directories(directories, model_names, destination_base)