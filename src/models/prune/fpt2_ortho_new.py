#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import torch
import pickle
import random
import sys
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import torch.nn.functional as F
import math
from tabulate import tabulate


import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import torch.nn as nn
from torch.optim import AdamW

import sys
sys.path.append(
    os.path.join(
        os.getcwd(),
        "src/models/"
    )
)   # Very hacky but the imports are annoying otherwise
from src.models.gpt2 import GPT2
from src.core import Config


device = torch.device("cuda")


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

class FPT2InfoTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.target_edge_sparsity = kwargs.pop('target_edge_sparsity', 0.0)
        self.start_edge_sparsity = kwargs.pop('start_edge_sparsity', 0.0)
        self.target_layer_sparsity = kwargs.pop('target_layer_sparsity', 0.0)
        self.start_layer_sparsity = kwargs.pop('start_layer_sparsity', 0.0)
        if "num_edge_sparsity_warmup_steps" in kwargs:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_edge_sparsity_warmup_steps')
        else:
            self.num_edge_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', 0)
        if "num_layer_sparsity_warmup_steps" in kwargs:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_layer_sparsity_warmup_steps')
        else:
            self.num_layer_sparsity_warmup_steps = kwargs.pop('num_sparsity_warmup_steps', self.num_edge_sparsity_warmup_steps)
        _ = kwargs.pop('num_sparsity_warmup_steps', None)
        self.warmup_type = kwargs.pop('warmup_type', 'linear')
        self.gpt2_model = kwargs.pop('gpt2_model', None)
        if self.gpt2_model is not None:
            self.gpt2_model.to(device)
        self.skip_layer_loss_if_higher_sparsity = kwargs.pop('skip_layer_loss_if_higher_sparsity', False)
        self.device_count = torch.cuda.device_count()
        self.use_truncated_kl_loss = kwargs.pop('use_truncated_kl_loss', False)
        self._train_acc_buffer = []
        self.k_for_truncated_kl = kwargs.pop('k_for_truncated_kl', 5)   
        self.margins = kwargs.pop('margins', {'set': 0.5, 'order': 0.0})
        self.set_loss_lambda = kwargs.pop('set_loss_lambda', 1.0)
        self.order_loss_lambda = kwargs.pop('order_loss_lambda', 1.0)
        self.save_total_limit = 2
        self.best_active_edges = float("inf")
        self.best_active_nodes = float("inf")
        super().__init__(*args, **kwargs)

    def get_current_edge_target_sparsity(self, global_step):
        if global_step < self.num_edge_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_edge_sparsity + (self.target_edge_sparsity - self.start_edge_sparsity) * 
                    global_step / self.num_edge_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_edge_sparsity) + (math.log(1 - self.target_edge_sparsity) - 
                    math.log(1 - self.start_edge_sparsity)) * global_step / self.num_edge_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_edge_sparsity
        
    def get_current_layer_target_sparsity(self, global_step):
        if global_step < self.num_layer_sparsity_warmup_steps:
            if self.warmup_type == 'linear':
                return (
                    self.start_layer_sparsity + (self.target_layer_sparsity - self.start_layer_sparsity) * 
                    global_step / self.num_layer_sparsity_warmup_steps
                )
            elif self.warmup_type == 'logarithmic':
                log_one_minus_sparsity = math.log(1 - self.start_layer_sparsity) + (math.log(1 - self.target_layer_sparsity) - 
                    math.log(1 - self.start_layer_sparsity)) * global_step / self.num_layer_sparsity_warmup_steps
                return 1 - math.exp(log_one_minus_sparsity)
            else:
                raise ValueError(f'Unknown warmup type: {self.warmup_type}')
        else:
            return self.target_layer_sparsity

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     # 1) unpack and reshape just like before
    #     start_idxes = inputs.pop("start_idxes")     # (B,)
    #     end_idxes   = inputs.pop("end_idxes")       # (B,)
    #     _           = inputs.pop("labels")          # (B, L) -- ignored here
    #     _           = inputs.pop("corr_input_ids")  # (B, L) -- still not used
    #     input_ids   = inputs.pop("input_ids")       # (B, num_sys, L, d_in=57)

    #     bsz, num_sys, L, d_in = input_ids.shape
    #     input_ids = torch.from_numpy(input_ids).to(device).float()
    #     input_ids = input_ids.view(bsz * num_sys, L, d_in)
    #     new_bsz = input_ids.shape[0]

    #     start_idxes = torch.from_numpy(start_idxes).to(device).long()
    #     end_idxes   = torch.from_numpy(end_idxes).to(device).long()
    #     start_idxes = torch.repeat_interleave(start_idxes, num_sys)
    #     end_idxes   = torch.repeat_interleave(end_idxes,   num_sys)

    #     # 2) split context vs. true‑next
    #     print("input_ids:", input_ids[:, :L, :][0])
    #     context   = input_ids[:, :L-1, :]
    #     print("context:", context[0])
    #     true_full = input_ids[:, L-1, :]
    #     print("true_full:", true_full[0])
    #     true_next = true_full[..., -5:]                           # (B⋅num_sys, 5)
    #     print("true_next:", true_next[0])

    #     # 3) student forward pass
    #     _, out_dict  = model.predict_step({"current": context})
    #     preds_full   = out_dict["preds"]                          # (B⋅num_sys, L-1, 5)
    #     pred_next    = preds_full[:, -1, :]                       # (B⋅num_sys, 5)

    #     # 4) teacher forward (no grad)
    #     with torch.no_grad():
    #         _, out_dict_ref = self.gpt2_model.predict_step({"current": context})
    #         preds_full_ref  = out_dict_ref["preds"]
    #     pred_next_ref = preds_full_ref[:, -1, :]

    #     # 5) MSE on the payload dims
    #     true_next = true_next.to(device).float()
    #     print("true_next:", true_next[0])
    #     print("pred_next:", pred_next[0])
    #     print("pred_next_ref:", pred_next_ref[0])
    #     mse_loss  = F.mse_loss(pred_next, true_next)

    #     # 6) KL‑divergence between student & teacher
    #     log_p_s = F.log_softmax(pred_next,    dim=-1)
    #     log_p_t = F.log_softmax(pred_next_ref,dim=-1)
    #     kl_full = F.kl_div(log_p_s, log_p_t, reduction="batchmean", log_target=True)

    #     def truncated_kl(x, y, k=None):
    #         if k is None:
    #             k = min(5, y.size(-1))
    #         _, idx = torch.topk(y, k=k, dim=-1)
    #         x_k = x.gather(-1, idx)
    #         y_k = y.gather(-1, idx)
    #         return F.kl_div(
    #             F.log_softmax(x_k, dim=-1),
    #             F.log_softmax(y_k, dim=-1),
    #             reduction="batchmean",
    #             log_target=True
    #         )

    #     kl_loss = truncated_kl(pred_next, pred_next_ref) if self.use_truncated_kl_loss else kl_full

    #     # 7) your sparsity regs
    #     reg_edge = out_dict.get("edge_loss", 0) or 0
    #     reg_node = out_dict.get("node_loss", 0) or 0

    #     # 8) total loss
    #     total_loss = mse_loss + kl_loss + reg_edge + reg_node

    #     # —— NEW: compute 5‑way classification accuracy —— 
    #     pred_classes = torch.argmax(pred_next, dim=-1)           # (B⋅num_sys,)
    #     true_classes = torch.argmax(true_next, dim=-1)           # (B⋅num_sys,)
    #     accuracy     = (pred_classes == true_classes).float().mean()
    #     out_dict["accuracy"] = accuracy
        
    #     # self._train_acc_buffer.append(accuracy.item())
    #     # if self.state.global_step % self.args.logging_steps == 0 and self._train_acc_buffer:
    #     #     avg_acc = sum(self._train_acc_buffer) / len(self._train_acc_buffer)
    #     #     self._train_acc_buffer.clear()
            
    #     self._train_acc_buffer.append(accuracy.item())

    #     # 9) package everything for logging & HF Trainer
    #     out_dict.update({
    #         "loss":      total_loss,
    #         "mse_loss":  mse_loss,
    #         "kl_loss":   kl_loss,
    #         "kl_full":   kl_full,
    #         "trunc_kl":  kl_loss if self.use_truncated_kl_loss else torch.tensor(0.0),
    #         "edge_loss": reg_edge,
    #         "node_loss": reg_node,
    #     })

    #     # carry forward sparsity metrics
    #     for k in ("active_edges","total_edges","active_nodes","total_nodes",
    #             "lambda_edges_1","lambda_edges_2","lambda_nodes_1","lambda_nodes_2"):
    #         out_dict[k] = out_dict.get(k, -1)

    #     # logs = {
    #     #     "mse_loss":      mse_loss.item(),
    #     #     "kl_loss":       kl_loss.item(),
    #     #     "active_edges":  out_dict["active_edges"],
    #     #     "total_edges":   out_dict["total_edges"],
    #     #     "active_nodes":  out_dict["active_nodes"],
    #     #     "total_nodes":   out_dict["total_nodes"],
    #     #     "lambda_edges_1":out_dict["lambda_edges_1"],
    #     #     "lambda_edges_2":out_dict["lambda_edges_2"],
    #     #     "lambda_nodes_1":out_dict["lambda_nodes_1"],
    #     #     "lambda_nodes_2":out_dict["lambda_nodes_2"],
    #     #     "accuracy":      accuracy.item(),
    #     #     "avg_accuracy":  avg_acc,
    #     # }
    #     # self.log(logs)
        
    #     logs = {
    #         "mse_loss":      mse_loss.item(),
    #         "kl_loss":       kl_loss.item(),
    #         "active_edges":  out_dict["active_edges"],
    #         "total_edges":   out_dict["total_edges"],
    #         "active_nodes":  out_dict["active_nodes"],
    #         "total_nodes":   out_dict["total_nodes"],
    #         "lambda_edges_1":out_dict["lambda_edges_1"],
    #         "lambda_edges_2":out_dict["lambda_edges_2"],
    #         "lambda_nodes_1":out_dict["lambda_nodes_1"],
    #         "lambda_nodes_2":out_dict["lambda_nodes_2"],
    #         "accuracy":      accuracy.item(),
    #     }

    #     # every logging_steps, compute & log average accuracy, then clear buffer
    #     if self.state.global_step % self.args.logging_steps == 0 and self._train_acc_buffer:
    #         avg_acc = sum(self._train_acc_buffer) / len(self._train_acc_buffer)
    #         logs["avg_accuracy"] = avg_acc
    #         self._train_acc_buffer.clear()

    #     self.log(logs)

    #     # (optional pretty‐table print, unchanged)
    #     table_data = [["Global Step", self.state.global_step]]
    #     table_data += [[k, f"{v: .4f}"] for k,v in logs.items()]
    #     table = tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty")
    #     if self.accelerator is None or self.accelerator.is_main_process:
    #         logger.info(f"\n{table}")

    #     return (total_loss, out_dict) if return_outputs else total_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1) unpack and reshape
        start_idxes     = inputs.pop("start_idxes")     # (B,)
        end_idxes       = inputs.pop("end_idxes")       # (B,)
        _               = inputs.pop("labels")          # (B, L) -- ignored
        corr_input_ids  = inputs.pop("corr_input_ids")  # (B, num_sys, L, d_in)
        input_ids       = inputs.pop("input_ids")       # (B, num_sys, L, d_in)

        bsz, num_sys, L, d_in = input_ids.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # flatten both real and corrupted batches
        input_ids      = torch.from_numpy(input_ids).to(device).float().view(bsz * num_sys, L, d_in)
        corr_input_ids = torch.from_numpy(corr_input_ids).to(device).float().view(bsz * num_sys, L, d_in)

        # repeat indices to match flattened batch
        start_idxes = torch.repeat_interleave(
            torch.from_numpy(start_idxes).to(device).long(), num_sys
        )
        end_idxes   = torch.repeat_interleave(
            torch.from_numpy(end_idxes).to(device).long(),   num_sys
        )

        # 2) build context vs next-token
        context      = input_ids[:, :L-1, :]      # (B*num_sys, L-1, d_in)
        corr_context = corr_input_ids[:, :L-1, :] # same shape for corrupted

        # 3) get the corrupted writer-states from teacher (no grad)
        embeds_corr = self.gpt2_model._read_in(corr_context)  # [B⋅num_sys, L-1, emb_dim]

        # 2) one no-grad forward *directly* through the backbone to grab writer_states
        with torch.no_grad():
            teacher_out = self.gpt2_model._backbone(
                inputs_embeds=embeds_corr,
                output_writer_states=True,   # this flag makes FPT2Model return writer_states
                corr_x=None 
            )
            corr_x = teacher_out.writer_states

        _, out_dict = model.predict_step(
            {"current": context},
            corr_x=corr_x,
            target_edge_sparsity=self.get_current_edge_target_sparsity(self.state.global_step),
            target_node_sparsity=self.get_current_layer_target_sparsity(self.state.global_step),
        )
        preds_full = out_dict["preds"]           # (B*num_sys, L-1, 5)
        pred_next  = preds_full[:, -1, :]        # (B*num_sys, 5)

        # 5) teacher forward on *real* context for KL target
        with torch.no_grad():
            _, out_dict_ref = self.gpt2_model.predict_step({"current": context})
            preds_full_ref = out_dict_ref["preds"]
        pred_next_ref = preds_full_ref[:, -1, :]

        # 6) compute losses
        true_next = input_ids[:, -1, -5:].to(device)   # the 5-dim payload
        mse_loss  = F.mse_loss(pred_next, true_next)

        # # KL or truncated KL
        # log_p_s = F.log_softmax(pred_next,    dim=-1)
        # log_p_t = F.log_softmax(pred_next_ref,dim=-1)
        # kl_full = F.kl_div(log_p_s, log_p_t, reduction="batchmean", log_target=True)

        # if self.use_truncated_kl_loss:
        #     def truncated_kl(x, y, k=5):
        #         _, idx = torch.topk(y, k=k, dim=-1)
        #         xk, yk = x.gather(-1, idx), y.gather(-1, idx)
        #         return F.kl_div(F.log_softmax(xk, dim=-1),
        #                         F.log_softmax(yk, dim=-1),
        #                         reduction="batchmean", log_target=True)
        #     kl_loss = truncated_kl(pred_next, pred_next_ref)
        # else:
        #     kl_loss = kl_full

        # sparsity regs
        reg_edge = out_dict["edge_loss"]
        
        if self.skip_layer_loss_if_higher_sparsity and out_dict["model_node_sparsity"] > out_dict["target_node_sparsity"]:
            reg_node = 0
        else:
            reg_node = out_dict["node_loss"]
        reg_loss = reg_edge + reg_node

        # # top-k ranking loss
        # def topk_ranking_loss(s_logits, t_logits, k, margins, lambdas, delta_kl=0.0):
        #     B, V = t_logits.shape
        #     topk_idx = torch.topk(t_logits, k, dim=-1).indices
        #     pos = s_logits.gather(-1, topk_idx)
        #     mask = torch.ones_like(t_logits, dtype=torch.bool)
        #     mask.scatter_(-1, topk_idx, False)
        #     neg = s_logits.masked_select(mask).view(B, V - k)

        #     set_margin   = margins['set'] - (pos.unsqueeze(-1) - neg.unsqueeze(1))
        #     loss_set     = torch.clamp(set_margin, min=0).mean()
        #     diff         = pos.unsqueeze(2) - pos.unsqueeze(1)
        #     rank_mask    = torch.triu(torch.ones(k, k, device=device, dtype=torch.bool), 1)
        #     order_margin = margins['order'] - diff.masked_select(rank_mask)
        #     loss_order   = torch.clamp(order_margin, min=0).mean()

        #     total = lambdas['set'] * loss_set + lambdas['order'] * loss_order
        #     if delta_kl:
        #         total = total + delta_kl * F.kl_div(
        #             F.log_softmax(s_logits, dim=-1),
        #             F.log_softmax(t_logits, dim=-1),
        #             reduction='batchmean', log_target=True
        #         )
        #     return total, loss_set, loss_order

        # k_rank = min(self.k_for_truncated_kl, pred_next.size(-1))
        # ranking_loss, set_loss, order_loss = topk_ranking_loss(
        #     pred_next, pred_next_ref,
        #     k=k_rank,
        #     margins=self.margins,
        #     lambdas={'set': self.set_loss_lambda, 'order': self.order_loss_lambda}
        # )

        # 7) total
        total_loss = 1000 * mse_loss + reg_loss #+ ranking_loss + kl_loss

        # 8) compute accuracy
        pred_cls = pred_next.argmax(dim=-1)
        true_cls = true_next.argmax(dim=-1)
        accuracy = (pred_cls == true_cls).float().mean()
        out_dict["accuracy"] = accuracy
        self._train_acc_buffer.append(accuracy.item())

        # 9) assemble out_dict & logs
        out_dict.update({
            "loss":         total_loss,
            "mse_loss":     mse_loss,
            # "kl_loss":      kl_loss,
            # "kl_full":      kl_full,
            "reg_loss":     reg_loss.detach().cpu().item(),
            "edge_loss":    reg_edge.detach().cpu().item(),
            "node_loss":    reg_node.detach().cpu().item(),
            # "ranking_loss": ranking_loss,
            # "set_loss":     set_loss,
            # "order_loss":   order_loss,
        })

        logs = {
            "mse_loss":     mse_loss.item(),
            # "kl_loss":      kl_loss.item(),
            # "ranking_loss": ranking_loss.item(),
            # "set_loss":     set_loss.item(),
            # "order_loss":   order_loss.item(),
            "accuracy":     accuracy.item(),
            "reg_loss":     reg_loss.detach().cpu().item(),
            "edge_loss":    reg_edge.detach().cpu().item(),
            "node_loss":    reg_node.detach().cpu().item(),
            "active_edges": out_dict.get("active_edges", -1),
            "total_edges":  out_dict.get("total_edges",  -1),
            "active_nodes": out_dict.get("active_nodes", -1),
            "total_nodes":  out_dict.get("total_nodes",  -1),
        }
        if self.state.global_step % self.args.logging_steps == 0 and self._train_acc_buffer:
            avg_acc = sum(self._train_acc_buffer) / len(self._train_acc_buffer)
            logs["avg_accuracy"] = avg_acc
            self._train_acc_buffer.clear()

        self.log(logs)

        # 10) pretty-table print
        table_data = [["Global Step", self.state.global_step]]
        for k, v in logs.items():
            table_data.append([k, f"{v: .4f}"])
        if getattr(self, "accelerator", None) is None or self.accelerator.is_main_process:
            logger.info("\n" + tabulate(table_data, headers=["Metric","Value"], tablefmt="pretty"))

        return (total_loss, out_dict) if return_outputs else total_loss

    def log(self, logs: dict, *args, **kwargs):
        """
        1) Delegate to HF’s logging & checkpoint logic (which respects save_total_limit=2).
        2) Once past warmup, if `active_edges` or `active_nodes` dips below its previous minimum,
           immediately torch.save(self.model.state_dict()) into a dedicated subfolder.
        """
        super().log(logs, *args, **kwargs)

        # step  = self.state.global_step
        # warm = getattr(self, "num_edge_sparsity_warmup_steps", 0)

        # if step <= warm:
        #     return

        # ae = logs.get("active_edges")
        # if ae is not None and ae < self.best_active_edges:
        #     self.best_active_edges = ae
        #     out = os.path.join(self.args.output_dir, f"best_active_edges_step_{step}")
        #     os.makedirs(out, exist_ok=True)
        #     torch.save(self.model.state_dict(), os.path.join(out, "state_dict.pt"))

        # an = logs.get("active_nodes")
        # if an is not None and an < self.best_active_nodes:
        #     self.best_active_nodes = an
        #     out = os.path.join(self.args.output_dir, f"best_active_nodes_step_{step}")
        #     os.makedirs(out, exist_ok=True)
        #     torch.save(self.model.state_dict(), os.path.join(out, "state_dict.pt"))

@dataclass
class DataTrainingArguments:
    dataset_path: Optional[str] = field(
        default="./identity_data/prune/",
        metadata={"help": "The path to the directory with the JSON files of the task."},
    )
    train_split: Optional[str] = field(
        default="train",
        metadata={"help": "The split to use for training."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    start_edge_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial edge sparsity of the model."}
    )
    target_edge_sparsity: Optional[float] = field(
        default=0.99,
        metadata={"help": "The target edge sparsity of the model."}
    )
    start_layer_sparsity: Optional[float] = field(
        default=0.0,
        metadata={"help": "The initial layer sparsity of the model."}
    )
    target_layer_sparsity: Optional[float] = field(
        default=0.80,
        metadata={"help": "The target layer sparsity of the model."}
    )
    stop_optimizing_layer_if_higher_sparsity: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to stop optimizing the layer sparsity if it is higher than the target."}
    )
    num_sparsity_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "The number of steps to reach the target sparsity."}
    )
    edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_edge_learning_rate: Optional[float] = field(
        default=1e-2,
        metadata={"help": "The learning rate for the regularization term."}
    )
    reg_layer_learning_rate: Optional[float] = field(
        default=1,
        metadata={"help": "The learning rate for the regularization term."}
    )
    warmup_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The type of warmup to use for the regularization term."}
    )
    with_embedding_nodes: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to include the embedding nodes"}
    )
    disable_linear_reg_term: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable the linear regularization term."}
    )
    disable_node_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to disable node loss."}
    )
    use_truncated_kl_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use truncated KL loss."}
    )
    use_two_after: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use two after."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    initialize_from: str = field(
        default="/scratch/users/dhruvgautam/models/models--sultan-daniels--TFs_do_KF_ICL_ident_med_GPT2_experiment/snapshots/f94c23e0e6a3c5c36cc04e005356cfa3ee007072/checkpoints/step=16000.ckpt",
        metadata={"help": "The model to initialize from."},
    )

def format_instance(instance, split):
    if isinstance(instance, dict) and "min_steps" in instance:
        return {
            "tokens": instance["tokens"],
            "split": split,
            "min_steps": instance["min_steps"],
        }
    else:
        return {
            "tokens": instance,
            "split": split,
        }

def load_datasets(dataset_path, max_train_samples, max_eval_samples, train_split="train"):
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    if "validation" not in dataset:
        assert max_eval_samples is not None, "Validation set is missing! (val)"
        assert max_train_samples is not None, "Validation set is missing! (train)"
        dataset = DatasetDict({
            train_split: dataset[train_split].select(range(max_train_samples)),
            "validation": dataset[train_split].select(range(max_train_samples, max_train_samples+max_eval_samples)),
        })
    else:
        if max_train_samples is not None and max_train_samples < len(dataset[train_split]):
            dataset[train_split] = dataset[train_split].select(range(max_train_samples))
        if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset

class DataCollatorGP:
    def __init__(
        self, 
        max_length,
        use_two_after,
    ):
        self.max_length = max_length 
        self.use_two_after = use_two_after

    def __call__(self, examples):
        input_ids_all = []
        corr_input_ids_all = []
        labels_all = []
        start_idxes = []
        end_idxes = []
        
        for example in examples:
            seq = example["sequence"]
            corr_seq = example["corr_sequence"]
            one_after_prediction = example["1_after_prediction"]
            corr_one_after_prediction = example["corr_1_after_prediction"]
            if self.use_two_after:
                two_after_prediction = example["2_after_prediction"]
                corr_two_after_prediction = example["corr_2_after_prediction"]
            # three_after_prediction = example["3_after_prediction"]
            # corr_three_after_prediction = example["corr_3_after_prediction"]
            
            l = len(seq)
            one_pred = np.expand_dims(one_after_prediction, axis=1)
            corr_one_pred = np.expand_dims(corr_one_after_prediction, axis=1)
            
            # preds = [seq, one_pred]
            # corr_preds = [corr_seq, corr_one_pred]
            input_ids = np.concatenate([seq, one_pred], axis=1)
            corr_input_ids = np.concatenate([corr_seq, corr_one_pred], axis=1)
            if self.use_two_after:
                two_pred = np.expand_dims(two_after_prediction, axis=1)
                corr_two_pred = np.expand_dims(corr_two_after_prediction, axis=1)
                input_ids = np.concatenate([seq, one_pred, two_pred], axis=1)
                corr_input_ids = np.concatenate([corr_seq, corr_one_pred, corr_two_pred], axis=1)
            # three_pred = np.expand_dims(three_after_prediction, axis=1)
            # corr_three_pred = np.expand_dims(corr_three_after_prediction, axis=1)

            labels = input_ids.copy()
            labels[:l] = -100
            # labels[labels == self.tokenizer.pad_token_id] = -100 # not sure if we need to pad our inputs in this setup

            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)
            
            # first_pad = (input_ids == self.tokenizer.pad_token_id).nonzero()[0]
            if self.use_two_after:
                start_idxes.append(l)
                end_idxes.append(l+1)
            else:
                start_idxes.append(l-1)
                end_idxes.append(l)
        
        return {
            "input_ids":      np.stack(input_ids_all, axis=0),      # shape (batch, 64, feat)
            "corr_input_ids": np.stack(corr_input_ids_all, axis=0),
            "labels":         np.stack(labels_all, axis=0),
            "start_idxes":    np.array(start_idxes, dtype=np.int64), # shape (batch,)
            "end_idxes":      np.array(end_idxes, dtype=np.int64),
        }       

def eval_fn(eval_pred): 
    # logits, target_edge_sparsity, target_layer_sparsity, model_edge_sparsity, model_layer_sparsity, reg_edge_loss, reg_layer_loss, kl_loss = eval_pred.predictions
    # if len(model_edge_sparsity.shape) > 0:
    #     model_edge_sparsity = model_edge_sparsity[0].item()
    #     model_layer_sparsity = model_layer_sparsity[0].item()
    #     target_edge_sparsity = target_edge_sparsity[0].item()
    #     target_layer_sparsity = target_layer_sparsity[0].item()
    # else:
    #     model_edge_sparsity = model_edge_sparsity.item()
    #     model_layer_sparsity = model_layer_sparsity.item()
    #     target_edge_sparsity = target_edge_sparsity.item()
    #     target_layer_sparsity = target_layer_sparsity.item()
    
    # predictions = np.argmax(logits, axis=-1)[:, :-1]
    # labels = eval_pred.label_ids[:, 1:]

    # eval_mask = (labels != -100).astype(int)
    # predictions = predictions * eval_mask
    # labels = labels * eval_mask
    
    # correct = (predictions == labels).all(axis=1)
    # accuracy = correct.sum().item() / correct.shape[0]
    
    # kl_loss = kl_loss.mean().item()
    # reg_edge_loss = reg_edge_loss.mean().item()
    # reg_layer_loss = reg_layer_loss.mean().item()
    
    return {
        # "eval_accuracy": accuracy,
        # "model_edge_sparsity": model_edge_sparsity,
        # "model_layer_sparsity": model_layer_sparsity,
        # "target_edge_sparsity": target_edge_sparsity,
        # "target_layer_sparsity": target_layer_sparsity,
        # "eval_kl_loss": kl_loss,
        # "eval_reg_edge_loss": reg_edge_loss,
        # "eval_reg_layer_loss": reg_layer_loss,
    }

@torch.no_grad()
def load_avg_activations(model, path, device):
    avg_activations = pickle.load(open(path, "rb"))
    model.load_captured_activations(avg_activations.to(device))
    
def freeze_all_except_pruning_params(model):
    for n, p in model.named_parameters():
        if 'log_alpha' in n or 'sparsity_lambda' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

def get_optimizers(model, edges_lr, layers_lr, reg_edges_lr, reg_layers_lr, num_training_steps, warmup_steps=0, disable_node_loss=False):
    optimizer_1_group = []
    optimizer_2_group = []
    optimizer_3_group = []
    optimizer_4_group = []

    for n, p in model.named_parameters():
        if 'write_log_alpha' in n:
            optimizer_3_group.append(p)
        elif 'read_log_alpha' in n:
            optimizer_1_group.append(p)
        elif 'sparsity_lambda_edge' in n:
            optimizer_2_group.append(p)
        elif ('sparsity_lambda_node' in n) and (not disable_node_loss):
            optimizer_4_group.append(p)
    
    optimizer = AdamW(
        [
            {
                'params': optimizer_1_group,
                'lr': edges_lr,
            },
            {
                'params': optimizer_2_group,
                'maximize': True,
                'lr': reg_edges_lr,
            },
            {
                'params': optimizer_3_group,
                'lr': layers_lr,
            },
            {
                'params': optimizer_4_group,
                'maximize': True,
                'lr': reg_layers_lr,
            } 
        ],
        lr=edges_lr
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    return optimizer, scheduler

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_datasets(data_args.dataset_path, data_args.max_train_samples, data_args.max_eval_samples, train_split=data_args.train_split)
    n_train = len(raw_datasets["train"])
    
    # model = FPT2LMHeadModel.from_pretrained(
    #     model_args.initialize_from,
    #     with_embedding_nodes=data_args.with_embedding_nodes,
    #     disable_linear_regularization_term=data_args.disable_linear_reg_term,
    # )
    # gpt2_model = FPT2LMHeadModel.from_pretrained(
    #     "gpt2",
    #     with_embedding_nodes=data_args.with_embedding_nodes,
    # ).to("cuda")
    model_name_or_path = "/scratch/users/dhruvgautam/models/models--sultan-daniels--TFs_do_KF_ICL_ortho_med_GPT2_experiment/snapshots/76bbc4fdd910adef1de36fc3e03828f23913f816/checkpoints/step=105000.ckpt"
    config = Config()
    checkpoint = torch.load(model_name_or_path, map_location="cuda")
    model = GPT2.load_from_checkpoint(model_name_or_path, n_dims_in=config.n_dims_in, n_positions=250, n_embd=128,
                                use_pos_emb=True, map_location=device, strict=False).eval().to(device)
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()},
        strict=False
    )
    
    gpt2_model = GPT2.load_from_checkpoint(model_name_or_path, n_dims_in=config.n_dims_in, n_positions=250, n_embd=128,
                                use_pos_emb=True, map_location=device, strict=False).eval().to(device)
    gpt2_model.load_state_dict(
        {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()},
        strict=False
    )
    
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # tokenizer.pad_token = tokenizer.eos_token
    
    freeze_all_except_pruning_params(model)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

    if training_args.do_eval:
        # We don't have a validation dataset, so we'll just use the test dataset.
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

    # Data collator
    collator = DataCollatorGP(
        max_length=data_args.max_seq_length,
        use_two_after=data_args.use_two_after
    )
    
    optimizers = get_optimizers(
        model, 
        edges_lr=data_args.edge_learning_rate,
        layers_lr=data_args.layer_learning_rate,
        reg_edges_lr=data_args.reg_edge_learning_rate,
        reg_layers_lr=data_args.reg_layer_learning_rate,
        num_training_steps=training_args.max_steps,
        warmup_steps=training_args.warmup_steps,
        disable_node_loss=data_args.disable_node_loss,
    )

    # Initialize our Trainer
    trainer = FPT2InfoTrainer(
        model=model,
        gpt2_model=gpt2_model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=eval_fn,
        optimizers=optimizers,
        start_edge_sparsity=data_args.start_edge_sparsity,
        target_edge_sparsity=data_args.target_edge_sparsity,
        start_layer_sparsity=data_args.start_layer_sparsity,
        target_layer_sparsity=data_args.target_layer_sparsity,
        skip_layer_loss_if_higher_sparsity=data_args.stop_optimizing_layer_if_higher_sparsity,
        num_sparsity_warmup_steps=data_args.num_sparsity_warmup_steps,
        warmup_type=data_args.warmup_type,
        use_truncated_kl_loss=data_args.use_truncated_kl_loss
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint,
        )
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    kwargs = {"finetuned_from": "gpt-2"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()