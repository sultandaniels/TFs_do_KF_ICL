import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import random

# =============================================================================
# 1. Load the pickle file that contains your multi-system traces.
# =============================================================================
filename = '/scratch/users/dhruvgautam/TFs_do_KF_ICL/ortho_data/data/interleaved_traces_ortho_ident_C_state_dim_5_num_sys_haystack_5.pkl'
with open(filename, 'rb') as f:
    file_dict = pickle.load(f)

# Assumed keys as provided:
multi_sys_ys = file_dict["multi_sys_ys"]
tok_seg_lens_per_config = file_dict["tok_seg_lens_per_config"]
sys_choices_per_config = file_dict["sys_choices_per_config"]

# For clarity, print an example shape:
# For example, the first system in the first sequence:
example_shape = np.array(multi_sys_ys[0, 0, :, :]).shape  # Expected: (73, 57)
print("Shape of multi_sys_ys[0,0,:,:]:", example_shape)
print("Length along segments (expected 73):", len(multi_sys_ys[0, 0, :, :]))
print("Length per segment (expected 57):", len(multi_sys_ys[0, 0, :, :][0]))

# =============================================================================
# 2. Rearrange and split the configurations into training, validation, and test sets.
#
#    Your data has shape (num_systems, 1000, 73, 57) with the 1000 sequences along
#    the second axis. We transpose so that each configuration (a sequence) is along axis 0.
# =============================================================================
multi_sys_ys = np.array(multi_sys_ys)  # Ensure it's a NumPy array

# Transpose from (num_systems, 1000, 73, 57) to (1000, num_systems, 73, 57)
config_data = np.transpose(multi_sys_ys, (1, 0, 2, 3))

# Now the first dimension is the 1000 sequences.
num_configs = config_data.shape[0]
print("Total number of configurations (sequences):", num_configs)
print("Full shape of config_data:", config_data.shape)

# Define split ratios.
train_ratio = 0.8
val_ratio = 0.1
# test_ratio automatically becomes: 1 - train_ratio - val_ratio

train_size = int(num_configs * train_ratio)
val_size = int(num_configs * val_ratio)
test_size = num_configs - train_size - val_size
print("Train/Val/Test split sizes:", train_size, val_size, test_size)

# Split the configurations along axis 0.
train_data = config_data[:train_size]      # shape: (train_size, num_systems, 73, 57)
val_data   = config_data[train_size:train_size + val_size]
test_data  = config_data[train_size + val_size:]

# =============================================================================
# 3. (Optional) Define a corruption function for a configuration.
#
# This function replaces one randomly chosen segment from one system with the
# corresponding segment from a different system, preserving the shape (num_systems, 73, 57).
# =============================================================================
def corrupt_configuration(config_array):
    """
    Given a configuration array of shape (num_systems, num_segments, feature_dim),
    replaces one randomly chosen segment from one system with the corresponding
    segment from a different system.
    """
    corrupted = config_array.copy()
    num_systems, num_segments, feature_dim = corrupted.shape

    # Choose a random system and a random segment to corrupt.
    system_to_replace = random.randint(0, num_systems - 1)
    segment_to_replace = random.randint(0, num_segments - 1)
    
    # Choose a different system to take the replacement segment.
    candidate_systems = list(range(num_systems))
    candidate_systems.remove(system_to_replace)
    replacement_system = random.choice(candidate_systems)
    
    # Replace the segment.
    corrupted[system_to_replace, segment_to_replace, :] = config_array[replacement_system, segment_to_replace, :]
    
    return corrupted

# Create "corrupted" versions for each split.
train_data_corr = np.array([corrupt_configuration(cfg) for cfg in train_data])
val_data_corr   = np.array([corrupt_configuration(cfg) for cfg in val_data])
test_data_corr  = np.array([corrupt_configuration(cfg) for cfg in test_data])

# =============================================================================
# 4. Create a torch Dataset that prepares examples with the desired keys.
#
# The returned dictionary for an example will have:
#
#   - "sequence"              : first 62 segments from the original data
#   - "1_after_prediction"    : the 63rd segment (index 62)
#   - "2_after_prediction"    : the 64th segment (index 63)
#   - "3_after_prediction"    : the 65th segment (index 64)
#
# and similarly for the corrupted version with keys prefixed by "corr_".
# =============================================================================
class MultiSysSplitDataset(Dataset):
    def __init__(self, originals, corrupted):
        """
        originals: NumPy array with shape (N, num_systems, 73, feature_dim)
        corrupted: NumPy array with the same shape as originals.
        """
        self.originals = originals
        self.corrupted = corrupted

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        # Convert the data to torch tensors.
        orig = torch.tensor(self.originals[idx], dtype=torch.float)  # shape: (num_systems, 73, feature_dim)
        corr = torch.tensor(self.corrupted[idx], dtype=torch.float)

        return {
            # For original data:
            "sequence": orig[:, :62, :],                 # first 62 segments
            "1_after_prediction": orig[:, 62, :],          # 63rd segment
            "2_after_prediction": orig[:, 63, :],          # 64th segment
            "3_after_prediction": orig[:, 64, :],          # 65th segment

            # For corrupted data:
            "corr_sequence": corr[:, :62, :],
            "corr_1_after_prediction": corr[:, 62, :],
            "corr_2_after_prediction": corr[:, 63, :],
            "corr_3_after_prediction": corr[:, 64, :]
        }

# Create datasets using the new dataset class.
train_dataset = MultiSysSplitDataset(train_data, train_data_corr)
val_dataset   = MultiSysSplitDataset(val_data, val_data_corr)
test_dataset  = MultiSysSplitDataset(test_data, test_data_corr)

# =============================================================================
# 5. Create DataLoaders for training and evaluation.
# =============================================================================
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

# =============================================================================
# 6. (Optional) Iterate over a batch to verify the shapes and key setup.
# =============================================================================
for batch in train_loader:
    # Each entry below is of shape:
    #   sequence: (batch_size, num_systems, 62, feature_dim)
    #   1_after_prediction, 2_after_prediction, 3_after_prediction: (batch_size, num_systems, feature_dim)
    print("Batch sequence shape:", batch["sequence"].shape)
    print("Batch 1_after_prediction shape:", batch["1_after_prediction"].shape)
    print("Batch 2_after_prediction shape:", batch["2_after_prediction"].shape)
    print("Batch 3_after_prediction shape:", batch["3_after_prediction"].shape)
    print("Batch corr_sequence shape:", batch["corr_sequence"].shape)
    print("Batch corr_1_after_prediction shape:", batch["corr_1_after_prediction"].shape)
    # Break after one batch for verification.
    break

# =============================================================================
# 7. Use these DataLoaders in your training/evaluation loop.
#
# For example:
#
#   for batch in val_loader:
#       inputs = batch["corr_sequence"].to(device)
#       targets = batch["sequence"].to(device)
#       # Use the after_prediction entries as needed for your prediction tasks.
#       # e.g., pred1 = model(inputs); loss = loss_fn(pred1, batch["1_after_prediction"].to(device))
#       ...
#
# This setup lets you access:
#
#   seq = example["sequence"]
#   corr_seq = example["corr_sequence"]
#   prediction = example["1_after_prediction"]
#   corr_prediction = example["corr_1_after_prediction"]
#
# with the sequence being the first 62 segments and the prediction keys corresponding to the next 3 segments.
# =============================================================================
