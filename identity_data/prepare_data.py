import pickle
import numpy as np
import random
from datasets import Dataset, DatasetDict

# =============================================================================
# 1. Load the pickle file that contains your multi-system traces.
# =============================================================================
input_filename = '/scratch/users/dhruvgautam/TFs_do_KF_ICL/identity_data/data/interleaved_traces_ident_ident_C_state_dim_5_num_sys_haystack_5.pkl'
with open(input_filename, 'rb') as f:
    file_dict = pickle.load(f)

# Keys: multi_sys_ys, tok_seg_lens_per_config, sys_choices_per_config
multi_sys_ys = file_dict["multi_sys_ys"]
tok_seg_lens_per_config = file_dict["tok_seg_lens_per_config"]
sys_choices_per_config = file_dict["sys_choices_per_config"]

# For example verification (optional):
example_shape = np.array(multi_sys_ys[0, 0, :, :]).shape  # expected (73, 57)
print("Shape of multi_sys_ys[0,0,:,:]:", example_shape)
print("Number of segments (expected 73):", len(multi_sys_ys[0, 0, :, :]))
print("Segment length (expected 57):", len(multi_sys_ys[0, 0, :, :][0]))

# =============================================================================
# 2. Rearrange configurations & split into training, validation, and test sets.
#
#    The original data is assumed to have shape: (num_systems, 1000, 73, 57).
#    We transpose it so that each configuration (a sequence) is along axis 0.
# =============================================================================
multi_sys_ys = np.array(multi_sys_ys)  # ensure a NumPy array
# Transpose: (num_systems, 1000, 73, 57) -> (1000, num_systems, 73, 57)
config_data = np.transpose(multi_sys_ys, (1, 0, 2, 3))
num_configs = config_data.shape[0]
print("Total number of configurations (sequences):", num_configs)
print("Full shape of config_data:", config_data.shape)

# Define split ratios:
train_ratio = 0.8
val_ratio = 0.1
train_size = int(num_configs * train_ratio)
val_size = int(num_configs * val_ratio)
test_size = num_configs - train_size - val_size
print("Train/Val/Test split sizes:", train_size, val_size, test_size)

# Split the configurations along the first axis.
train_data = config_data[:train_size]      # shape: (train_size, num_systems, 73, 57)
val_data   = config_data[train_size:train_size + val_size]
test_data  = config_data[train_size + val_size:]

# =============================================================================
# 3. (Optional) Define a corruption function for a configuration.
#
# This function replaces one randomly chosen segment of one system with the
# corresponding segment from another system (keeping the same shape).
# =============================================================================
def corrupt_configuration(config_array):
    """
    Given a configuration array with shape (num_systems, num_segments, feature_dim),
    replace one randomly chosen segment from one system with the corresponding segment
    from a different system.
    """
    corrupted = config_array.copy()
    num_systems, num_segments, feature_dim = corrupted.shape

    # Select a random system and segment
    system_to_replace = random.randint(0, num_systems - 1)
    segment_to_replace = random.randint(0, num_segments - 1)

    # Choose a replacement system (ensuring it’s different)
    candidate_systems = list(range(num_systems))
    candidate_systems.remove(system_to_replace)
    replacement_system = random.choice(candidate_systems)

    # Replace the segment
    corrupted[system_to_replace, segment_to_replace, :] = config_array[replacement_system, segment_to_replace, :]
    return corrupted

# Create corrupted versions for each split.
train_data_corr = np.array([corrupt_configuration(cfg) for cfg in train_data])
val_data_corr   = np.array([corrupt_configuration(cfg) for cfg in val_data])
test_data_corr  = np.array([corrupt_configuration(cfg) for cfg in test_data])

# =============================================================================
# 4. Convert each configuration (original and corrupted) into a dictionary format.
#
# Each example will have keys:
#
#   - "sequence"             : first 62 segments from the original configuration.
#   - "1_after_prediction"   : the 63rd segment.
#   - "2_after_prediction"   : the 64th segment.
#   - "3_after_prediction"   : the 65th segment.
#
#   And similarly for the corrupted version (prefixed with "corr_").
# =============================================================================
def convert_config_to_dict(orig, corr):
    """
    Converts the original configuration and its corrupted version into a dictionary.
    Uses the first 62 segments as the sequence and segments 63-65 as the targets.
    """
    return {
        "sequence": orig[:, :62, :].tolist(),
        "1_after_prediction": orig[:, 62, :].tolist(),
        "2_after_prediction": orig[:, 63, :].tolist(),
        "3_after_prediction": orig[:, 64, :].tolist(),
        "corr_sequence": corr[:, :62, :].tolist(),
        "corr_1_after_prediction": corr[:, 62, :].tolist(),
        "corr_2_after_prediction": corr[:, 63, :].tolist(),
        "corr_3_after_prediction": corr[:, 64, :].tolist()
    }

# Build lists of examples for each split.
train_examples = [convert_config_to_dict(orig, corr)
                  for orig, corr in zip(train_data, train_data_corr)]
val_examples   = [convert_config_to_dict(orig, corr)
                  for orig, corr in zip(val_data, val_data_corr)]
test_examples  = [convert_config_to_dict(orig, corr)
                  for orig, corr in zip(test_data, test_data_corr)]

# =============================================================================
# 5. Create a DatasetDict and save to disk.
#
# This step converts the dictionaries into HuggingFace Dataset objects and 
# bundles them into a DatasetDict for later use in your training/evaluation.
# =============================================================================
processed = DatasetDict({
    "train": Dataset.from_list(train_examples),
    "validation": Dataset.from_list(val_examples),
    "test": Dataset.from_list(test_examples)
})

# Specify the output directory where the processed dataset will be saved.
out_dir = '/scratch/users/dhruvgautam/TFs_do_KF_ICL/identity_data/prune/'  # Change this path to your desired output directory.
processed.save_to_disk(out_dir)
print(f"Processed dataset saved to: {out_dir}")
