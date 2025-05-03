#!/usr/bin/env python
import os
import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
import torch.nn.functional as F
from scipy.stats import kendalltau

from src.models.gpt2 import GPT2  
from src.core import Config

class bcolors:
    OKBLUE = '\033[94m'
    ENDC   = '\033[0m'

def info(text):
    print(f"{bcolors.OKBLUE}{text}{bcolors.ENDC}")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model_name_or_path",
        default="/scratch/users/dhruvgautam/TFs_do_KF_ICL/data/runs/two-after-identity-wo_node_loss-elr0.8-llr0.8-relr0.8-rllr0.8-es0.995-ns0.72-t3000/checkpoint-3000",
        help="Path to your fine-tuned checkpoint")
    parser.add_argument("-d","--data_path",
        default="./identity_data/prune/",
        help="Where you called `save_to_disk`")
    parser.add_argument("-s","--split", default="test",
        choices=["train","validation","test"])
    parser.add_argument("-b","--batch_size", default=16, type=int)
    parser.add_argument("-D","--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("-o","--out_path", default=None,
        help="Where to dump JSON summary")
    args = parser.parse_args()
    if args.out_path=="None": args.out_path=None

    device = torch.device(args.device)
    info(f"[i] Loading model → {args.model_name_or_path}")
    config = Config()
    model = GPT2.load_from_checkpoint(args.model_name_or_path, n_dims_in=config.n_dims_in, n_positions=250, n_embd=128,
                                use_pos_emb=True, map_location=device, strict=False).eval().to(device)
    model.eval()

    info(f"[i] Loading dataset → {args.data_path}[{args.split}]")
    ds = load_from_disk(args.data_path)[args.split]

    def collate(batch):
        # each ex has keys "sequence","corr_sequence","2_after_prediction"
        seq = torch.tensor([ex["sequence"] for ex in batch],
                           dtype=torch.float32, device=device)
        one_after = torch.tensor([ex["1_after_prediction"] for ex in batch],
                           dtype=torch.float32, device=device)
        corr= torch.tensor([ex["corr_sequence"] for ex in batch],
                           dtype=torch.float32, device=device)
        tgt_full = torch.tensor([ex["2_after_prediction"] for ex in batch],
                                dtype=torch.float32, device=device)
        seq = torch.cat([seq, one_after], dim=2)
        return seq, corr, tgt_full

    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collate, shuffle=False)

    info("[i] Running 2-step-ahead inference (last 5 dims)…")
    mse_list = []
    tau_list = []

    for seq, corr_seq, tgt_full in tqdm(loader):
        # seq: [B, num_sys, 63, feat_dim]
        B, N, T, D = seq.shape

        # flatten batch×systems for predict_step
        ctx = seq.reshape(B*N, T, D)
        corr_ctx = corr_seq.reshape(B*N, T, D)

        # ---- student forward ----
        _, out = model.predict_step({"current": ctx, "corr_x": corr_ctx})
        preds_full = out["preds"]           # [B*N, T, 5]
        pred2      = preds_full[:, -1, :]   # take the last time step → shape [B*N, 5]

        # ---- ground truth: last-5 dims of the 2_after vector ----
        tgt2_full  = tgt_full.reshape(B*N, D)
        tgt2       = tgt2_full[:, -5:]      # [B*N, 5]

        # ---- metrics ----
        mse = F.mse_loss(pred2, tgt2, reduction="mean")
        mse_list.append(mse.item())

        # Kendall’s τ per example
        p_np = pred2.cpu().numpy()
        t_np = tgt2.cpu().numpy()
        for p_vec, t_vec in zip(p_np, t_np):
            tau, _ = kendalltau(p_vec, t_vec)
            tau_list.append(tau)

    mean_mse = sum(mse_list)/len(mse_list)
    mean_tau = sum(tau_list)/len(tau_list)

    info(f"[i] ▶ Mean MSE over last-5 dims: {mean_mse:.6f}")
    info(f"[i] ▶ Mean Kendall τ    : {mean_tau:.6f}")

    if args.out_path:
        info(f"[i] Saving summary to {args.out_path}")
        summary = {
            "mean_mse_last5": mean_mse,
            "mean_kendall_tau": mean_tau,
            "batches":       len(mse_list),
            "examples_eval": len(ds),
        }
        with open(args.out_path, "w") as fp:
            json.dump(summary, fp, indent=4)

if __name__=="__main__":
    main()
