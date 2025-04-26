import os
from huggingface_hub import HfApi, HfFolder, hf_hub_download

api = HfApi(token=os.getenv("HF_TOKEN"))


#download a file from the hub
target_path = "/data/shared/ICL_Kalman_Experiments/train_and_test_data/"
repo_id = "sultan-daniels/train_and_test_data"
file_name = "val_ortho_haar_ident_C_state_dim_5.pkl"
subfolder = "ortho_haar"
file_path = hf_hub_download(
    repo_id=repo_id,
    filename=file_name,
    repo_type="dataset",
    revision="main",
    subfolder=subfolder,
    local_dir=target_path,
)
print(f"Downloaded {file_name} to {file_path}")


# api.upload_folder(
#     folder_path="/data/shared/ICL_Kalman_Experiments/train_and_test_data",
#     repo_id="sultan-daniels/train_and_test_data",
#     repo_type="dataset",
# )

# from huggingface_hub import create_repo

# # Repository details
# repo_id = "sultan-daniels/try2"
# local_repo = "../try2"  # Absolute path recommended
# # Create a new repo
# create_repo(
#     repo_id=repo_id,
#     repo_type="model",
#     private=True,
#     exist_ok=True
# )
# api.update_repo_settings(
#     repo_id=repo_id,
#     gated="manual"
# )

# # Create checkpoint directory
# checkpoint_path = os.path.join(local_repo, "checkpoint-2000", "pytorch_model.bin")
# os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
# # Create a dummy file
# with open(checkpoint_path, "wb") as f:
#     f.write(os.urandom(1024))  # Write 1KB of random data

# # Upload directly to repo
# api.upload_file(
#     path_or_fileobj=checkpoint_path,
#     path_in_repo="checkpoint-2000/pytorch_model.bin",
#     repo_id=repo_id,
#     repo_type="model"
# )

