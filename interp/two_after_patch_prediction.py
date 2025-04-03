import os
import json
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import pickle
from transformers import GPT2Model, GPT2LMHeadModel, AutoTokenizer
import sys
import numpy as np
from src.models import GPT2
from src.core import Config

timestamp = datetime.datetime.now().strftime("%d%H%M")
from interp.utils_gpt2 import cache_activations, generate_substitute_layer_single_logits

device = torch.device("cuda")

def swap_token_activations_logits(model, input_strings, args_b, token_a, token_b):
    prefix = "" # setup to see if 2 after can be predicted so this should just contain: special open token, first token
    
    final_list = input_strings
    swap_seq = [6, 7, 8, 9, 10]
    full_seq = len(input_strings[0]) #full frozen sequence

    seq = list(range(0, len(input_strings[0])))
    if not final_list:
        print("No valid inputs to process.")
        return
    print(f"Length of final_list: {len(final_list)}")

    layers = list(range(args_b.n_layers))
    module_list_a = [model.transformer.h[i] for i in layers]

    activation_cache_a_out = []
    activation_cache_b_out = []
    
    for idx in layers:
        module_str = f"model.transformer.h[{idx}]"
        try:
            cached_out_a = cache_activations(
                model=model,
                module_list_or_str=[module_str],
                cache_input_output='output',
                inputs=[final_list[0]],
                batch_size=args_b.batch_size,
                token_idx=seq,
            )[0]
            cached_out_b = cache_activations(
                model=model,
                module_list_or_str=[module_str],
                cache_input_output='output',
                inputs=[final_list[1]],
                batch_size=args_b.batch_size,
                token_idx=seq,
            )[0]
            activation_cache_a_out.append(cached_out_a)
            activation_cache_b_out.append(cached_out_b)
        except Exception as e:
            print(f"Error caching activations for layer {idx}: {e}")
            return

    log_prob_a_changes = []
    log_prob_b_changes = []

    for num_tokens_to_swap in range(0, len(swap_seq) + 1):
        activation_cache_swapped_out = []

        for layer_idx, (layer_out_a, layer_out_b) in enumerate(zip(activation_cache_a_out, activation_cache_b_out)):
            swapped_layer_out = layer_out_a.clone()
            try:
                for j in range(num_tokens_to_swap):
                    swapped_layer_out[:, j+5] = layer_out_b[:, j+5]
            except IndexError:
                print(f"Token indices out of bounds at swap {j}.")
                return
            activation_cache_swapped_out.append(swapped_layer_out)

        try:
            sub_out_logits = generate_substitute_layer_single_logits(
                model,
                final_list[0],
                module_list_a,
                activation_cache_swapped_out,
                'output',
                token_idx=seq,
                max_new_tokens=30
            )
            sub_out_log_probs = F.log_softmax(sub_out_logits, dim=-1)
            next_token_log_probs = sub_out_log_probs[:, -1, :]
            log_prob_a_changes.append(next_token_log_probs[:, token_a].cpu().numpy())
            log_prob_b_changes.append(next_token_log_probs[:, token_b].cpu().numpy())
        except Exception as e:
            print(f"Error generating substitute logits for swap {num_tokens_to_swap}: {e}")
            return

    print("Token-by-token swapping complete.")
    return log_prob_a_changes, log_prob_b_changes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/dhruv_gautam/models/models--sultan-daniels--TFs_do_KF_ICL_ortho_med_GPT2_checkpoints/snapshots/824c3034ec025999d7bc2923335142b19152ab71")
    parser.add_argument("--layer-skip", type=int, default=3)
    parser.add_argument("--batch-size", "-bs", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="post_emerge.ckpt")
    return parser.parse_args()

def plot_results(log_prob_a_changes, log_prob_b_changes, token_a, token_b):
    x_labels = [str(i) for i in range(len(log_prob_a_changes))]
    plt.figure(figsize=(12, 8))
    
    plt.plot(range(len(log_prob_a_changes)), log_prob_a_changes, marker='o', label=f"Token A ({token_a})")
    plt.plot(range(len(log_prob_b_changes)), log_prob_b_changes, marker='x', label=f"Token B ({token_b})")
    
    for i, (a, b) in enumerate(zip(log_prob_a_changes, log_prob_b_changes)):
        plt.text(i, a.item(), f'{a.item():.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i, b.item(), f'{b.item():.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Number of Tokens Swapped')
    plt.ylabel('Log Probability')
    plt.title('Log Probability Changes Across Token Swaps')
    plt.legend()
    plt.xticks(range(len(x_labels)), x_labels)
    plt.grid(True)
    plt.savefig(f'../figures/two_after_patch_prediction_{timestamp}.png')
    plt.show()


def setup_model(args):
    model_name_or_path = args.model
    checkpoint_path = f"{model_name_or_path}/{args.checkpoint}"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    # Load model architecture
    # n_dims_in = int(ny + (2*max_sys_trace) + 2) if multi_sys_trace else ny #input dimension is the observation dimension #input dimension is the observation dimension + special token parentheses + special start token + payload identifier
    config = Config()
    if "gpt" in model_name_or_path.lower():
        model = GPT2.load_from_checkpoint(model_name_or_path + "/post_emerge.ckpt", n_dims_in=config.n_dims_in, n_positions=250, n_embd=72,
                                use_pos_emb=True, map_location=device, strict=True).eval().to(device)
        #model = GPT2LMHeadModel.from_pretrained("gpt2")

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()},
        strict=False
    )

    if "gpt-j" in model_name_or_path.lower() or "GPT" in model_name_or_path:
        module_str_dict = {
            "layer": "model.transformer.h[{layer_idx}]",
            "attn": "model.transformer.h[{layer_idx}].attn.o_proj",
        }
        n_layers = len(model.transformer.h)
    else:
        raise ValueError(f"Unknown model architecture for {model_name_or_path}")

    args.module_str_dict = module_str_dict
    args.n_layers = n_layers

    return model, tokenizer

if __name__ == "__main__":
    # model_path = "/data/dhruv_gautam/models/models--sultan-daniels--TFs_do_KF_ICL_ortho_med_GPT2_checkpoints/snapshots/824c3034ec025999d7bc2923335142b19152ab71/post_emerge.ckpt"
    # checkpoint = torch.load(model_path, map_location="cuda")
    # print(checkpoint.keys())
    # model = GPT2LMHeadModel.from_pretrained("gpt2")
    # model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}, strict=False)
    # # model = GPT2LMHeadModel.from_pretrained("gpt2")  
    # # model.load_state_dict(torch.load(model_path, map_location="cuda"))
    # tokenizer_path = "/data/dhruv_gautam/models/models--sultan-daniels--TFs_do_KF_ICL_ortho_med_GPT2_checkpoints/snapshots/824c3034ec025999d7bc2923335142b19152ab71/"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    args = get_args()
    model, tokenizer = setup_model(args)
    model.to(device)

    dataset_path = "/data/dhruv_gautam/models/models--sultan-daniels--TFs_do_KF_ICL_ortho_med_GPT2_experiment/snapshots/6255a948d411c6e3b0182ce3f308ce25c9c5d725/data/val_ortho_ident_C_state_dim_5.pkl"
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded. Type: {type(dataset)}")
    if isinstance(dataset, dict):  # If it's a dictionary
        print(f"Keys: {list(dataset.keys())}")
        sequence_prompts = dataset.get("input_sequences", [])  # Use .get to avoid KeyError
    elif isinstance(dataset, list) and dataset:  # If it's a non-empty list
        print(f"Sample Entry: {dataset[0]}")
        if isinstance(dataset[0], dict):  # Check if list elements are dictionaries
            possible_keys = dataset[0].keys()
            print(f"Possible keys in entries: {possible_keys}")
            sequence_prompts = [entry.get("obs", None) for entry in dataset]
        else:
            sequence_prompts = dataset  # If not a dictionary, treat as raw sequence data
    else:
        sequence_prompts = []


    filename = "/data/dhruv_gautam/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/data/interleaved_traces_ortho_ident_C_state_dim_5_num_sys_haystack_1.pkl"

    with open(filename, 'rb') as f:
        file_dict = pickle.load(f)

    multi_sys_ys = file_dict["multi_sys_ys"]
    tok_seg_lens_per_config = file_dict["tok_seg_lens_per_config"]
    sys_choices_per_config = file_dict["sys_choices_per_config"]

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    print("First row of multi_sys_ys[0,0,:,:]:\n")
    print(len(multi_sys_ys[0, 0, :, :][0])) # 57

    print(len(multi_sys_ys[0, 0, :, :])) # 25

    for config_idx, (seg_lengths, sys_choices) in enumerate(zip(tok_seg_lens_per_config, sys_choices_per_config)):
        print(f"\nConfiguration {config_idx}:")
        # Print details for each segment
        for seg_idx, (length, sys_choice) in enumerate(zip(seg_lengths, sys_choices)):
            print(f"  Segment {seg_idx}: Length = {length}, System = {sys_choice}")
        
        # Determine if this configuration is interleaved
        unique_systems = set(sys_choices)
        if len(unique_systems) > 1:
            print("-> This trace is interleaved with systems:", unique_systems)
        else:
            print("-> This trace is not interleaved; it contains only system:", unique_systems.pop())

    
    haystack_prompts = [
        multi_sys_ys[0, 0, :, :],
        multi_sys_ys[0, 1, :, :]
    ]
    
    token_a = tokenizer.convert_tokens_to_ids("") # what it should predict
    token_b = tokenizer.convert_tokens_to_ids("") # what it should predict if swap interferes properly (all encodings on the same token)
    
    log_prob_a_changes, log_prob_b_changes = swap_token_activations_logits(
        model, haystack_prompts, args, token_a, token_b
    )
    
    plot_results(log_prob_a_changes, log_prob_b_changes, token_a, token_b)