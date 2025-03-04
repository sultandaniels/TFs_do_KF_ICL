import os
from huggingface_hub import HfApi, HfFolder

# Set the token as an environment variable
os.environ["HUGGINGFACE_HUB_TOKEN"] = "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"

from huggingface_hub import HfApi, HfFolder

HfFolder.save_token("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")
api = HfApi()

api.upload_large_folder(
    repo_id="sultan-daniels/TFs_do_KF_ICL_ortho_med_GPT2_experiment",
    repo_type="model",
    folder_path="/home/sultand/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000"
)