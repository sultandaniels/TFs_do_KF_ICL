import torch
import numpy as np
import pickle
from src.models.randomTransformer import RandomTransformerUnembedding
from src.core import Config
from torch.utils.data import DataLoader

# Load config
config = Config()
config.override("multi_sys_trace", True)
config.override("max_sys_trace", 25)
config.override("ny", 5)
config.override("n_dims_in", int(config.ny + (2 * config.max_sys_trace) + 2))  # 57
config.override("n_dims_out", 5)
config.override("n_positions", 250)
config.override("use_true_len", True)

n_dims_in = config.n_dims_in  # 57
n_dims_out = config.n_dims_out  # 5
n_positions = config.n_positions  # 250
n_embd = 128
n_layer = 12
n_head = 8

# Create the model
model = RandomTransformerUnembedding(
    n_dims_in=n_dims_in,
    n_positions=n_positions,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    n_dims_out=n_dims_out
)

print(f"Model created successfully!")
print(f"Input dimension: {n_dims_in}")
print(f"Output dimension: {n_dims_out}")
print(f"Embedding dimension: {n_embd}")

# Always use the real data file
real_data_path = 'src/DataRandomTransformer/train_interleaved_traces_ortho_haar_ident_C_multi_cut.pkl'
print(f"Loading real data from {real_data_path}")
with open(real_data_path, 'rb') as f:
    data = pickle.load(f)

multi_sys_ys = data['multi_sys_ys']  # shape: (1, 40000, 1, 251, 57)
N = np.prod(multi_sys_ys.shape[:-2])
seq_len = multi_sys_ys.shape[-2]
inputs = multi_sys_ys.reshape(N, seq_len, 57)
targets = inputs[..., -5:]

num_samples = min(1000, len(inputs))
print(f"Using {num_samples} samples for activation extraction")

activations_list = []
targets_list = []

print("Extracting activations from frozen backbone...")

with torch.no_grad():
    for i in range(num_samples):
        current_input = torch.from_numpy(inputs[i]).unsqueeze(0).float()  # [1, seq_len, n_dims_in]
        target_output = torch.from_numpy(targets[i]).unsqueeze(0).float()  # [1, seq_len, n_dims_out]
        embeds = model._read_in(current_input)
        backbone_output = model._backbone(inputs_embeds=embeds).last_hidden_state
        activations_list.append(backbone_output.cpu())
        targets_list.append(target_output.cpu())
        if i % 100 == 0:
            print(f"Processed {i}/{num_samples} samples")

all_activations = torch.cat(activations_list, dim=0)  # [num_samples, seq_len, n_embd]
all_targets = torch.cat(targets_list, dim=0)

print(f"\nExtracted activations shape: {all_activations.shape}")
print(f"Targets shape: {all_targets.shape}")

activation_data = {
    'activations': all_activations.numpy(),
    'targets': all_targets.numpy(),
    'n_embd': n_embd,
    'n_dims_out': n_dims_out,
    'using_real_data': True
}

with open('activation_dataset.pkl', 'wb') as f:
    pickle.dump(activation_data, f)

print(f"Saved activation dataset to 'activation_dataset.pkl'")
print(f"Using real data: True")

# Create a dataset class for the activations
class ActivationDataset(torch.utils.data.Dataset):
    def __init__(self, activations, targets):
        self.activations = torch.from_numpy(activations).float()
        self.targets = torch.from_numpy(targets).float()
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

# Test the dataset
dataset = ActivationDataset(all_activations.numpy(), all_targets.numpy())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"\nCreated activation dataset with {len(dataset)} samples")
print(f"Dataloader batch size: 32")

# Test the dataloader
print(f"\nTesting activation dataset...")
for batch_activations, batch_targets in dataloader:
    print(f"Batch activations shape: {batch_activations.shape}")
    print(f"Batch targets shape: {batch_targets.shape}")
    break

print(f"\nActivation extraction completed successfully!")
print(f"You can now train only the output layer using this pre-computed activation dataset.") 