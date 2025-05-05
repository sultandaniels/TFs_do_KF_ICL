import os
import json
import argparse

import torch
import sys
from src.models.gpt2 import GPT2    
from src.core import Config

try:
    from safetensors.torch import load_file as safe_load
except ImportError:
    raise ImportError("Please `pip install safetensors` to load .safetensors weights")

device = torch.device("cuda")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", "-w",
        type=str,
        required=True,
        help="Path to the .safetensors (or Lightning .ckpt) file"
    )
    parser.add_argument(
        "--out_path", "-o",
        type=str,
        default=None,
        help="Where to write the JSON edges (default: ./edges.json)"
    )
    parser.add_argument(
        "--with_embedding_nodes", "-e",
        action="store_true",
        help="Whether to include embedding nodes in the graph"
    )
    parser.add_argument(
        "--edge_sparsity", "-s",
        type=float,
        default=None,
        help="Target edge sparsity (if omitted, uses model.get_edge_sparsity())"
    )
    parser.add_argument(
        "--node_sparsity", "-n",
        type=float,
        default=None,
        help="Target node sparsity (if omitted, uses model.get_node_sparsity())"
    )
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.abspath("edges.json")
        print(f"Output path not specified. Saving to {args.out_path}.")

    return args

def load_weights(path, device):
    """
    Load either:
      - a .safetensors file
      - a Lightning .ckpt/.pth file (which wraps its weights under a 'state_dict' key)
      - a plain PyTorch state-dict saved to .pt
    Returns a state_dict with all tensors already on `device`.
    """
    ext = os.path.splitext(path)[1]
    if ext == ".safetensors":
        # load tensors on CPU…
        sd_cpu = safe_load(path)           # defaults to CPU
        # …then move them all onto GPU (or whatever device you passed)
        return {k: v.to(device) for k, v in sd_cpu.items()}

    elif ext == ".pt":
        # plain state_dict saved via torch.save(model.state_dict(), 'model.pt')
        sd_cpu = torch.load(path, map_location="cpu")
        if not isinstance(sd_cpu, dict):
            raise ValueError(f"Expected a dict in {path}, got {type(sd_cpu)}")
        return {k: v.to(device) for k, v in sd_cpu.items()}

    elif ext in (".ckpt", ".pth"):
        # Lightning checkpoint or other torch.save(...) wrapper
        ckpt = torch.load(path, map_location="cpu")
        raw = ckpt.get("state_dict", ckpt)
        if not isinstance(raw, dict):
            raise ValueError(f"Could not find 'state_dict' in {path}")
        return {k: v.to(device) for k, v in raw.items()}

    else:
        raise ValueError("Unrecognized extension for weights_path. Use .safetensors, .pt, .ckpt or .pth")



def main():
    args = parse_args()

    # instantiate your GPT2 LightningModule
    config = Config()
    model = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=250,
        n_embd=128,
        use_pos_emb=True
    ).eval().to(device)

    # load and massage the state dict
    raw_sd = load_weights(args.weights_path, device)
    # strip any "model." prefixes if present
    state_dict = {k.replace("model.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(state_dict, strict=False)

    # determine target sparsities
    target_edge_sp = args.edge_sparsity if args.edge_sparsity is not None else model._backbone.get_edge_sparsity()
    target_node_sp = args.node_sparsity if args.node_sparsity is not None else model._backbone.get_node_sparsity()

    # binary‐search threshold for edges
    l, r = 0.0, 1.0
    while r - l > 1e-5:
        mid = (l + r) / 2
        model._backbone.set_edge_threshold_for_deterministic(mid)
        if model._backbone.get_edge_sparsity() > target_edge_sp:
            r = mid
        else:
            l = mid

    # binary‐search threshold for nodes
    l2, r2 = 0.0, 1.0
    while r2 - l2 > 1e-1:
        mid = (l2 + r2) / 2
        model._backbone.set_node_threshold_for_deterministic(mid)
        if model._backbone.get_node_sparsity() > target_node_sp:
            r2 = mid
        else:
            l2 = mid

    # report and dump
    print("Overall edge sparsity:", model._backbone.get_effective_edge_sparsity().item())
    edges = model._backbone.get_edges()
    with open(args.out_path, "w") as f:
        json.dump(edges, f, indent=4)
    print(f"Edges written to {args.out_path}")

if __name__ == "__main__":
    main()
