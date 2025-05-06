#!/usr/bin/env python
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
    parser = argparse.ArgumentParser(description="Compute edge overlaps between two GPT2 sparsified circuits")
    parser.add_argument(
        "--weights_path1",
        type=str,
        required=True,
        help="First model weights file (.safetensors|.pt|.ckpt|.pth)"
    )
    parser.add_argument(
        "--weights_path2",
        type=str,
        required=True,
        help="Second model weights file (.safetensors|.pt|.ckpt|.pth)"
    )
    parser.add_argument(
        "--edge_sparsity", "-s",
        type=float,
        default=None,
        help="Target edge sparsity for clipping (default: use model.get_edge_sparsity())"
    )
    parser.add_argument(
        "--node_sparsity", "-n",
        type=float,
        default=None,
        help="Target node sparsity (unused here but parsed for consistency)"
    )
    return parser.parse_args()

def load_weights(path, device):
    ext = os.path.splitext(path)[1]
    if ext == ".safetensors":
        sd_cpu = safe_load(path)
        return {k: v.to(device) for k, v in sd_cpu.items()}
    elif ext == ".pt":
        sd_cpu = torch.load(path, map_location="cpu")
        if not isinstance(sd_cpu, dict):
            raise ValueError(f"Expected a dict in {path}, got {type(sd_cpu)}")
        return {k: v.to(device) for k, v in sd_cpu.items()}
    elif ext in (".ckpt", ".pth"):
        ckpt = torch.load(path, map_location="cpu")
        raw = ckpt.get("state_dict", ckpt)
        if not isinstance(raw, dict):
            raise ValueError(f"Could not find 'state_dict' in {path}")
        return {k: v.to(device) for k, v in raw.items()}
    else:
        raise ValueError("Unrecognized extension; use .safetensors, .pt, .ckpt or .pth")

def build_model(path, config, args):
    m = GPT2(
        n_dims_in=config.n_dims_in,
        n_positions=250,
        n_embd=128,
        use_pos_emb=True
    ).eval().to(device)
    raw_sd = load_weights(path, device)
    sd = {k.replace("model.", ""): v for k, v in raw_sd.items()}
    m.load_state_dict(sd, strict=False)
    return m

def extract_edges(model, target_sparsity=None):
    if target_sparsity is not None:
        lo, hi = 0.0, 1.0
        # binary search threshold to match target sparsity
        while hi - lo > 1e-3:
            mid = (lo + hi) / 2
            model._backbone.set_edge_threshold_for_deterministic(mid)
            if model._backbone.get_edge_sparsity() > target_sparsity:
                hi = mid
            else:
                lo = mid
    # get_edges() returns a JSON-serializable list of [src, dst]
    return model._backbone.get_edges()

def main():
    args = parse_args()
    config = Config()

    # Load both models
    m1 = build_model(args.weights_path1, config, args)
    m2 = build_model(args.weights_path2, config, args)

    # Unclipped edges (all non-zero)
    e1_unc = extract_edges(m1, target_sparsity=None)
    e2_unc = extract_edges(m2, target_sparsity=None)

    # Clipped edges (to each model's target sparsity)
    target_sp1 = args.edge_sparsity if args.edge_sparsity is not None else m1._backbone.get_edge_sparsity()
    target_sp2 = args.edge_sparsity if args.edge_sparsity is not None else m2._backbone.get_edge_sparsity()
    e1_cl = extract_edges(m1, target_sparsity=target_sp1)
    e2_cl = extract_edges(m2, target_sparsity=target_sp2)

    # Convert to sets of tuples
    s1_unc = {tuple(edge) for edge in e1_unc}
    s2_unc = {tuple(edge) for edge in e2_unc}
    s1_cl  = {tuple(edge) for edge in e1_cl}
    s2_cl  = {tuple(edge) for edge in e2_cl}

    overlap_unc = s1_unc & s2_unc
    overlap_cl  = s1_cl  & s2_cl

    # Print results
    print(f"Model1 unclipped edges: {len(s1_unc)}")
    print(f"Model2 unclipped edges: {len(s2_unc)}")
    print(f"Unclipped overlap edges: {len(overlap_unc)}")
    print()

    print(f"Model1 clipped edges (@ sparsity={target_sp1:.4f}): {len(s1_cl)}")
    print(f"Model2 clipped edges (@ sparsity={target_sp2:.4f}): {len(s2_cl)}")
    print(f"Clipped overlap edges: {len(overlap_cl)}")
    print()

    # Optionally, list the overlapping edges themselves
    print("Sample unclipped overlap (first 20):")
    for edge in list(overlap_unc)[:20]:
        print(" ", edge)
    print()

    print("Sample clipped overlap (first 20):")
    for edge in list(overlap_cl)[:20]:
        print(" ", edge)

if __name__ == "__main__":
    main()
