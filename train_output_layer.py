import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src.models.gpt2 import GPT2
from src.models.randomTransformer import RandomTransformerUnembedding
from src.core import Config

# Helper to load and preprocess data
def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    multi_sys_ys = data['multi_sys_ys']
    N = np.prod(multi_sys_ys.shape[:-2])
    seq_len = multi_sys_ys.shape[-2]
    inputs = multi_sys_ys.reshape(N, seq_len, 57)
    targets = inputs[..., -5:]
    return inputs, targets

# Load train and val data
print("Loading training data from train_interleaved_traces_ortho_haar_ident_C_multi_cut.pkl ...")
train_inputs, train_targets = load_data('src/DataRandomTransformer/train_interleaved_traces_ortho_haar_ident_C_multi_cut.pkl')
print(f"Loaded train inputs shape: {train_inputs.shape}")
print(f"Loaded train targets shape: {train_targets.shape}")

print("Loading validation data from val_interleaved_traces_ortho_haar_ident_C_haystack_len_1.pkl ...")
val_inputs, val_targets = load_data('src/DataRandomTransformer/val_interleaved_traces_ortho_haar_ident_C_haystack_len_1.pkl')
print(f"Loaded val inputs shape: {val_inputs.shape}")
print(f"Loaded val targets shape: {val_targets.shape}")

# Randomly sample 10% of samples for faster training
np.random.seed(42)  # For reproducibility

train_indices = np.random.choice(len(train_inputs), size=int(0.1 * len(train_inputs)), replace=False)
val_indices = np.random.choice(len(val_inputs), size=int(0.1 * len(val_inputs)), replace=False)

train_inputs = train_inputs[train_indices]
train_targets = train_targets[train_indices]
val_inputs = val_inputs[val_indices]
val_targets = val_targets[val_indices]

print(f"Using 10% of samples: {len(train_inputs)} training samples and {len(val_inputs)} validation samples")
print(f"Final train inputs shape: {train_inputs.shape}")
print(f"Final val inputs shape: {val_inputs.shape}")

# Create dataset class for activations
class ActivationDataset(Dataset):
    def __init__(self, activations, targets):
        self.activations = torch.FloatTensor(activations)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.targets[idx]

# Create datasets
train_dataset = ActivationDataset(train_inputs, train_targets)
val_dataset = ActivationDataset(val_inputs, val_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Model 1: RandomTransformerUnembedding (only output layer trainable)
def create_random_transformer_model(shared_weights=None):
    config = Config()
    n_dims_in = 57  # 5 + (2*25) + 2
    n_positions = 250
    n_embd = 128
    n_layer = 12
    n_head = 8
    n_dims_out = 5
    
    model = RandomTransformerUnembedding(
        n_dims_in=n_dims_in,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_dims_out=n_dims_out
    )
    if shared_weights is not None:
        model._read_in.load_state_dict(shared_weights['embedding'])
        model._backbone.load_state_dict(shared_weights['backbone'])
    return model

def create_gpt2_model(shared_weights=None):
    config = Config()
    n_dims_in = 57
    n_positions = 250
    n_embd = 128
    n_layer = 12
    n_head = 8
    n_dims_out = 5
    model = GPT2(
        n_dims_in=n_dims_in,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_dims_out=n_dims_out
    )
    if shared_weights is not None:
        model._read_in.load_state_dict(shared_weights['embedding'])
        model._backbone.load_state_dict(shared_weights['backbone'])
    return model

print("Initializing shared random weights for embedding and backbone...")
_temp_model = RandomTransformerUnembedding(
    n_dims_in=57, n_positions=250, n_embd=128, n_layer=12, n_head=8, n_dims_out=5
)
shared_weights = {
    'embedding': _temp_model._read_in.state_dict(),
    'backbone': _temp_model._backbone.state_dict()
}
del _temp_model

# Training function
def train_model(model, train_loader, test_loader, model_name, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_activations, batch_targets in train_loader:
            batch_activations = batch_activations.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            if model_name == "RandomTransformerUnembedding":
                # For RandomTransformerUnembedding, we need to create the full input structure
                batch_size, seq_len, n_embd = batch_activations.shape
                # Create dummy input data (since we're using pre-computed activations)
                dummy_input = torch.randn(batch_size, seq_len, 57).to(device)
                
                # Forward pass through RandomTransformerUnembedding using predict_step
                input_dict = {"current": dummy_input, "target": batch_targets}
                _, intermediate_dict = model.predict_step(input_dict)
                outputs = intermediate_dict["preds"]
            else:  # GPT2 model
                # For GPT2, we need to create the full input structure
                batch_size, seq_len, n_embd = batch_activations.shape
                # Create dummy input data (since we're using pre-computed activations)
                dummy_input = torch.randn(batch_size, seq_len, 57).to(device)
                
                # Forward pass through GPT2 using predict_step
                input_dict = {"current": dummy_input, "target": batch_targets}
                _, intermediate_dict = model.predict_step(input_dict)
                outputs = intermediate_dict["preds"]
            
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Testing
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_activations, batch_targets in test_loader:
                batch_activations = batch_activations.to(device)
                batch_targets = batch_targets.to(device)
                
                if model_name == "RandomTransformerUnembedding":
                    # For RandomTransformerUnembedding, we need to create the full input structure
                    batch_size, seq_len, n_embd = batch_activations.shape
                    dummy_input = torch.randn(batch_size, seq_len, 57).to(device)
                    input_dict = {"current": dummy_input, "target": batch_targets}
                    _, intermediate_dict = model.predict_step(input_dict)
                    outputs = intermediate_dict["preds"]
                else:  # GPT2 model
                    batch_size, seq_len, n_embd = batch_activations.shape
                    dummy_input = torch.randn(batch_size, seq_len, 57).to(device)
                    input_dict = {"current": dummy_input, "target": batch_targets}
                    _, intermediate_dict = model.predict_step(input_dict)
                    outputs = intermediate_dict["preds"]
                
                loss = criterion(outputs, batch_targets)
                test_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    return train_losses, test_losses

# Train both models
print("=" * 50)
print("TRAINING RANDOM TRANSFORMER UNEMBEDDING MODEL")
print("=" * 50)

model1 = create_random_transformer_model(shared_weights=shared_weights)
train_losses1, test_losses1 = train_model(model1, train_loader, test_loader, "RandomTransformerUnembedding", epochs=50)

print("\n" + "=" * 50)
print("TRAINING FULL GPT2 MODEL")
print("=" * 50)

model2 = create_gpt2_model(shared_weights=shared_weights)
train_losses2, test_losses2 = train_model(model2, train_loader, test_loader, "Full GPT2", epochs=50)

# Plotting
plt.figure(figsize=(12, 8))

# Plot training losses
plt.subplot(2, 2, 1)
plt.plot(train_losses1, label='RandomTransformerUnembedding', color='blue')
plt.plot(train_losses2, label='Full GPT2', color='red')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot test losses
plt.subplot(2, 2, 2)
plt.plot(test_losses1, label='RandomTransformerUnembedding', color='blue')
plt.plot(test_losses2, label='Full GPT2', color='red')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot final comparison
plt.subplot(2, 2, 3)
epochs = range(len(train_losses1))
plt.plot(epochs, train_losses1, 'b-', label='RandomTransformerUnembedding (Train)', alpha=0.7)
plt.plot(epochs, test_losses1, 'b--', label='RandomTransformerUnembedding (Test)', alpha=0.7)
plt.plot(epochs, train_losses2, 'r-', label='Full GPT2 (Train)', alpha=0.7)
plt.plot(epochs, test_losses2, 'r--', label='Full GPT2 (Test)', alpha=0.7)
plt.title('Training vs Test Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Final performance comparison
plt.subplot(2, 2, 4)
models = ['RandomTransformer\nUnembedding', 'Full GPT2']
final_train_losses = [train_losses1[-1], train_losses2[-1]]
final_test_losses = [test_losses1[-1], test_losses2[-1]]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, final_train_losses, width, label='Final Train Loss', alpha=0.8)
plt.bar(x + width/2, final_test_losses, width, label='Final Test Loss', alpha=0.8)
plt.xlabel('Model Type')
plt.ylabel('Loss')
plt.title('Final Performance Comparison')
plt.xticks(x, models)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final results
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"RandomTransformerUnembedding - Final Train Loss: {final_train_losses[0]:.6f}")
print(f"RandomTransformerUnembedding - Final Test Loss: {final_test_losses[0]:.6f}")
print(f"Full GPT2 - Final Train Loss: {final_train_losses[1]:.6f}")
print(f"Full GPT2 - Final Test Loss: {final_test_losses[1]:.6f}")

# Save the trained models
torch.save(model1.state_dict(), 'random_transformer_unembedding_model.pth')
torch.save(model2.state_dict(), 'full_gpt2_model.pth')
print("\nModels saved as 'random_transformer_unembedding_model.pth' and 'full_gpt2_model.pth'")
print("Plot saved as 'model_comparison.png'") 