from huggingface_hub import hf_hub_download

repo_id = "sultan-daniels/TFs_do_KF_ICL_ortho_med_GPT2_checkpoints"
filename = "step=42000.ckpt"
custom_path = "/data/dhruv_gautam/models"  

file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=custom_path)
print(f"Downloaded to: {file_path}")