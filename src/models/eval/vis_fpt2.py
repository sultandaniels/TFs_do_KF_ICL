import os
import json
import argparse

import torch
import sys
from src.models.gpt2 import GPT2
from src.core import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", "-c",
        type=str,
        required=True,
        help="Path to the Lightning .ckpt file"
    )
    parser.add_argument(
        "--out_path", "-o",
        type=str,
        default=None,
        help="Where to write the JSON edges (default: ./edges.json)"
    )
    parser.add_argument(
        "--with_embedding_nodes", "-w",
        action="store_true",
        help="Whether to include embedding nodes in the graph"
    )
    parser.add_argument(
        "--edge_sparsity", "-e",
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

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load Lightning checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    config = Config()

    # instantiate your GPT2 LightningModule
    model = GPT2.load_from_checkpoint(
        args.ckpt_path,
        n_dims_in=config.n_dims_in,
        n_positions=250,
        n_embd=128,
        use_pos_emb=True,
        map_location=device,
        strict=False
    ).eval().to(device)

    # strip the "model." prefix if present
    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=False)

    # determine target sparsities
    target_edge_sp = args.edge_sparsity if args.edge_sparsity is not None else model.get_edge_sparsity()
    target_node_sp = args.node_sparsity if args.node_sparsity is not None else model.get_node_sparsity()

    # binary‐search threshold for edges
    l, r = 0.0, 1.0
    while r - l > 1e-5:
        mid = (l + r) / 2
        model.set_edge_threshold_for_deterministic(mid)
        if model.get_edge_sparsity() > target_edge_sp:
            r = mid
        else:
            l = mid

    # binary‐search threshold for nodes
    l2, r2 = 0.0, 1.0
    while r2 - l2 > 1e-5:
        mid = (l2 + r2) / 2
        model.set_node_threshold_for_deterministic(mid)
        if model.get_node_sparsity() > target_node_sp:
            r2 = mid
        else:
            l2 = mid

    # report and dump
    print("Overall edge sparsity:", model.get_effective_edge_sparsity().item())
    edges = model.get_edges()
    with open(args.out_path, "w") as f:
        json.dump(edges, f, indent=4)
    print(f"Edges written to {args.out_path}")

if __name__ == "__main__":
    main()
