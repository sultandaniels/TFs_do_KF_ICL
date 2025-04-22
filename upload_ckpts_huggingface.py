import os
from huggingface_hub import HfApi, HfFolder, hf_hub_download

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="/data/shared/ICL_Kalman_Experiments/train_and_test_data",
    repo_id="sultan-daniels/train_and_test_data",
    repo_type="dataset",
)
