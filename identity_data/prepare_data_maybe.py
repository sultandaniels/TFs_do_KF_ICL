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

# These keys are assumed to be present from your description.
multi_sys_ys = file_dict["multi_sys_ys"]
tok_seg_lens_per_config = file_dict["tok_seg_lens_per_config"]
sys_choices_per_config = file_dict["sys_choices_per_config"]

# For clarity, let's print the shapes as you did:
# For example, the first system in the first configuration:
print("Shape of multi_sys_ys[0,0,:,:]:", np.array(multi_sys_ys[0,0,:,:]).shape)  # Expected: (73, 57)
print("Length along segments (expected 73):", len(multi_sys_ys[0,0,:,:]))              # 73
print("Length per segment (expected 57):", len(multi_sys_ys[0,0,:,:][0]))                 # 57

# =============================================================================
# 2. Split the configurations into training, validation, and test sets.
#    (Here we simply split along the first dimension: the number of configurations.)
# =============================================================================
num_configs = multi_sys_ys.shape[0]
print("Total number of configurations:", num_configs)

# Define split sizes (you can adjust these numbers as needed)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 1 - train_ratio - val_ratio

train_size = int(num_configs * train_ratio)
val_size = int(num_configs * val_ratio)
test_size = num_configs - train_size - val_size

print("Train/Val/Test split:", train_size, val_size, test_size)

# Note: multi_sys_ys is assumed to be a NumPy array. If not, wrap it with np.array().
multi_sys_ys = np.array(multi_sys_ys)

train_data = multi_sys_ys[:train_size]  # shape: (train_size, num_systems, 73, 57)
val_data   = multi_sys_ys[train_size:train_size + val_size]
test_data  = multi_sys_ys[train_size + val_size:]

# =============================================================================
# 3. Define a corruption function that produces a “corrupted” copy of a configuration.
#
# For each configuration (which is a set of outputs from several systems), we:
#   - Choose a random system index and a random segment index.
#   - Replace that segment from the chosen system with the segment from another (different) system.
#
# This operation preserves the overall shape:
#   (num_systems, 73, 57)
#
# (In your text-based example the corruption preserved token counts.)
# =============================================================================
def corrupt_configuration(config_array):
    """
    Given a configuration array of shape (num_systems, num_segments, feature_dim),
    replace one randomly chosen segment from one system with the same segment
    from a different system.
    """
    corrupted = config_array.copy()
    num_systems, num_segments, feature_dim = corrupted.shape

    # Choose a random system and a random segment to corrupt.
    system_to_replace = random.randint(0, num_systems - 1)
    segment_to_replace = random.randint(0, num_segments - 1)
    
    # Choose a different system from which to take the replacement segment.
    candidate_systems = list(range(num_systems))
    candidate_systems.remove(system_to_replace)
    replacement_system = random.choice(candidate_systems)
    
    # Perform the replacement; this ensures the corrupted trace has the same shape.
    corrupted[system_to_replace, segment_to_replace, :] = config_array[replacement_system, segment_to_replace, :]
    
    return corrupted

# Create "corrupted" versions for each split.
train_data_corr = np.array([corrupt_configuration(cfg) for cfg in train_data])
val_data_corr   = np.array([corrupt_configuration(cfg) for cfg in val_data])
test_data_corr  = np.array([corrupt_configuration(cfg) for cfg in test_data])

# =============================================================================
# 4. Create a torch Dataset to serve pairs: original and corrupted.
# =============================================================================
class MultiSysDataset(Dataset):
    def __init__(self, originals, corrupted):
        """
        originals: NumPy array with shape (N, num_systems, 73, 57)
        corrupted: NumPy array with same shape as originals.
        """
        self.originals = originals
        self.corrupted = corrupted

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        # Convert data to torch tensors.
        return {
            'original': torch.tensor(self.originals[idx], dtype=torch.float),
            'corrupted': torch.tensor(self.corrupted[idx], dtype=torch.float)
        }

train_dataset = MultiSysDataset(train_data, train_data_corr)
val_dataset   = MultiSysDataset(val_data, val_data_corr)
test_dataset  = MultiSysDataset(test_data, test_data_corr)

# =============================================================================
# 5. Create DataLoaders for training and evaluation.
# =============================================================================
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

# =============================================================================
# 6. (Optional) Example: iterate over the training loader and print the shapes.
# =============================================================================
for batch in train_loader:
    originals = batch['original']  # Shape: (batch_size, num_systems, 73, 57)
    corrupted = batch['corrupted']  # Same shape as originals.
    print("Batch originals shape:", originals.shape)
    print("Batch corrupted shape:", corrupted.shape)
    # Break after one batch for demonstration purposes.
    break

# =============================================================================
# 7. You can now pass these DataLoaders to your training/evaluation loop.
#
# For instance, suppose your model expects an input of shape (num_systems, 73, 57) 
# and you want to compare predictions on the corrupted inputs against the originals.
#
# (The interleaved_mse() function in your original snippet deals with combining
#  segments from different systems. Since you said to ignore interleaving, you can
#  use the raw shapes as provided.)
#
# Example (pseudo-code):
#
#   for batch in val_loader:
#       inputs = batch['corrupted'].to(device)
#       targets = batch['original'].to(device)
#       preds = model(inputs)
#       loss = loss_fn(preds, targets)
#       ...
#
# This setup mirrors your original template where the "corruption" is applied
# yet the array shapes remain consistent.
# =============================================================================
