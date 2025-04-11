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
from src.data_train import set_config_params


timestamp = datetime.datetime.now().strftime("%d%H%M")
from interp.utils_gpt2 import cache_activations, generate_substitute_layer_single_logits

device = torch.device("cuda")

model_name = "ident"

if model_name == "ident":
    valA = "ident"
    valC = "_ident_C"
    nx = 5

# model_name = "ortho"
# if model_name == "ortho":
#     valA = "ortho"
#     valC = "_ortho_C"
#     nx = 5


checkpoint_path = "/scratch/users/dhruvgautam/models/models--sultan-daniels--TFs_do_KF_ICL_ident_med_GPT2_experiment/snapshots/f94c23e0e6a3c5c36cc04e005356cfa3ee007072/checkpoints/step=16000.ckpt"
ckpt_step = 16000

# config = Config()
# output_dir, ckpt_dir, experiment_name = set_config_params(config, model_name)

# print(f"ckpt_dir: {ckpt_dir}")
# config.override("ckpt_path", ckpt_dir + f"/checkpoints/step={ckpt_step}.ckpt")
# print(f"ckpt_path: {config.ckpt_path}")

# num_gpu = len(config.devices)
# batch_size = config.batch_size
# print(f"Number of GPUs: {num_gpu}")
# print(f"Batch size: {batch_size}")
# print(f"Number of training examples: {ckpt_step*batch_size*num_gpu}")

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
    parser.add_argument("--model", type=str, default="/scratch/users/dhruvgautam/models/models--sultan-daniels--TFs_do_KF_ICL_ident_med_GPT2_experiment/snapshots/f94c23e0e6a3c5c36cc04e005356cfa3ee007072/checkpoints/step=16000.ckpt")
    parser.add_argument("--layer-skip", type=int, default=3)
    parser.add_argument("--batch-size", "-bs", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="step=16000.ckpt")
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

    # model = GPT2.load_from_checkpoint(config.ckpt_path,
    #                             n_dims_in=config.n_dims_in, n_positions=config.n_positions,
    #                             n_dims_out=config.n_dims_out, n_embd=config.n_embd,
    #                             n_layer=config.n_layer, n_head=config.n_head, use_pos_emb=config.use_pos_emb, map_location=device, strict=True).eval().to(device)
    # # Load model architecture
    # n_dims_in = int(ny + (2*max_sys_trace) + 2) if multi_sys_trace else ny #input dimension is the observation dimension #input dimension is the observation dimension + special token parentheses + special start token + payload identifier
    config = Config()
    if "gpt" in model_name_or_path.lower():
        model = GPT2.load_from_checkpoint(model_name_or_path, n_dims_in=config.n_dims_in, n_positions=250, n_embd=128,
                                use_pos_emb=True, map_location=device, strict=True).eval().to(device)
        #model = GPT2LMHeadModel.from_pretrained("gpt2")

    checkpoint = torch.load(model_name_or_path, map_location="cuda")
    model.load_state_dict(
        {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()},
        strict=False
    )

    if "GPT" in model_name_or_path:
        module_str_dict = {
            "layer": "model._backbone.h[{layer_idx}]",
            "attn": "model._backbone.h[{layer_idx}].attn.o_proj",
        }
        n_layers = len(model._backbone.h)
        print(n_layers)
        print(model._backbone.h[0].attn.c_proj)
    
    else:
        raise ValueError(f"Unknown model architecture for {model_name_or_path}")

    args.module_str_dict = module_str_dict
    args.n_layers = n_layers

    return model

def tf_preds(multi_sys_ys, model, device, config):
    with torch.no_grad():  # no gradients
        print(multi_sys_ys.shape[-2])
        I = np.take(multi_sys_ys, np.arange(multi_sys_ys.shape[-2] - 1), axis=-2)  # get the inputs (observations without the last one)

        print("before model.predict_step()")
        batch_shape = I.shape[:-2]
        print("batch_shape:", batch_shape)
        flattened_I = np.reshape(I, (np.prod(batch_shape), *I.shape[-2:]))
        print("flattened_I.shape:", flattened_I.shape)
        validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I), batch_size=config.test_batch_size)
        preds_arr = []  # Store the predictions for all batches
        for validation_batch in iter(validation_loader):
            _, flattened_preds_tf = model.predict_step({"current": validation_batch.to(device)})  # predict using the model
            preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
        print(preds_arr)
        print(len(preds_arr))
        print(len(preds_arr[0]))
        print(len(preds_arr[0][0]))
        preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                            (*batch_shape, 24, config.ny))  # Combine the predictions for all batches
        print("preds_tf.shape:", preds_tf.shape)
        preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf], axis=-2)  # concatenate the predictions
        ground_truth = np.take(multi_sys_ys, [multi_sys_ys.shape[-2] - 1], axis=-2)
        
        ground_truth_last5 = ground_truth[..., -5:]  # shape: (batch, 1, 5)
        
        # Extract the predicted last timestep from preds_tf (should be shape: (batch, 1, 5))
        predicted_last = preds_tf[:, -1:, :]
        
        # Calculate mean squared error (MSE) over the last 5 features
        mse = np.mean((ground_truth_last5 - predicted_last) ** 2)
        print("Ground truth (last timestep, last 5 features):")
        print(ground_truth)
        print(ground_truth_last5)
        print("Predicted last timestep (5 features):")
        print(predicted_last)
        print("MSE between ground truth and prediction for the last 5 features:", mse)
    return preds_tf

def interfered_mse(multi_sys_ys, model, device, config):
    with torch.no_grad():  # no gradients
        print(multi_sys_ys[0].shape[-2])
        
        I = np.take(multi_sys_ys[0], np.arange(multi_sys_ys[0].shape[-2] - 2), axis=-2)
        other_sys = np.take(multi_sys_ys[1], [multi_sys_ys[1].shape[-2] - 2], axis=-2)
        I = np.concatenate([I, other_sys], axis=-2)

        print("before model.predict_step()")
        batch_shape = I.shape[:-2]
        print("batch_shape:", batch_shape)
        flattened_I = np.reshape(I, (int(np.prod(batch_shape)), *I.shape[-2:]))
        print("flattened_I.shape:", flattened_I.shape)
        validation_loader = torch.utils.data.DataLoader(torch.from_numpy(flattened_I), batch_size=config.test_batch_size)
        preds_arr = []  # Store the predictions for all batches
        for validation_batch in iter(validation_loader):
            _, flattened_preds_tf = model.predict_step({"current": validation_batch.to(device)})  # predict using the model
            preds_arr.append(flattened_preds_tf["preds"].cpu().numpy())
        print(preds_arr)
        print(len(preds_arr))
        print(len(preds_arr[0]))
        print(len(preds_arr[0][0]))
        preds_tf = np.reshape(np.concatenate(preds_arr, axis=0),
                            (*batch_shape, 24, config.ny))  # Combine the predictions for all batches
        print("preds_tf.shape:", preds_tf.shape)
        preds_tf = np.concatenate([np.zeros_like(np.take(preds_tf, [0], axis=-2)), preds_tf], axis=-2)  # concatenate the predictions
        ground_truth_sys1 = np.take(multi_sys_ys[0], [multi_sys_ys[0].shape[-2] - 2], axis=-2)
        ground_truth_sys2 = np.take(multi_sys_ys[1], [multi_sys_ys[1].shape[-2] - 2], axis=-2)
        
        ground_truth_last5_sys1 = ground_truth_sys1[..., -5:]  # shape: (batch, 1, 5)
        ground_truth_last5_sys2 = ground_truth_sys2[..., -5:]  # shape: (batch, 1, 5)
        
        # Extract the predicted last timestep from preds_tf (should be shape: (batch, 1, 5))
        predicted_last = preds_tf[-1:, :]
        
        # Calculate mean squared error (MSE) over the last 5 features
        mse_sys1 = np.mean((ground_truth_last5_sys1 - predicted_last) ** 2)
        mse_sys2 = np.mean((ground_truth_last5_sys2 - predicted_last) ** 2)
        print("Ground truth (last timestep, last 5 features):")
        print(ground_truth_sys1)
        print(ground_truth_last5_sys1)
        print(ground_truth_sys2)
        print(ground_truth_last5_sys2)
        print("Predicted last timestep (5 features):")
        print(predicted_last)
        print("MSE between ground truth and prediction for the last 5 features:", mse_sys1)
        print("MSE between ground truth and prediction for the last 5 features:", mse_sys2)
    return preds_tf

if __name__ == "__main__":
    args = get_args()
    model = setup_model(args)
    model.to(device)

    dataset_path = "/scratch/users/dhruvgautam/TFs_do_KF_ICL/identity_data/data/val_ident_ident_C_state_dim_5.pkl"
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    
    filename = '/scratch/users/dhruvgautam/TFs_do_KF_ICL/identity_data/data/interleaved_traces_ident_ident_C_state_dim_5_num_sys_haystack_1.pkl'

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

    print("\nFirst row of multi_sys_ys[0,0,:,:]:\n")
    print(multi_sys_ys[0, 0, :, :])
    print("\nFirst row of multi_sys_ys[0,1,:,:]:\n")
    print(multi_sys_ys[0, 1, :, :])
    
    haystack_prompts = [
        multi_sys_ys[0, 0, :, :],
        multi_sys_ys[0, 1, :, :]
    ]
    
    predictions = []  # To store outputs for each prompt
    config = Config()

    # for i, prompt in enumerate(haystack_prompts):
    #     # Ensure the prompt has a batch dimension (if needed)
    #     # For example, if prompt.shape is (25, 57), convert it to (1, 25, 57)
    #     prompt_batched = np.expand_dims(prompt, axis=0)
        
    #     # Run inference for the single prompt
    #     pred = tf_preds(prompt_batched, model, device, config)
        
    #     # If you prefer to remove the batch dimension afterward, you can do so:
    #     pred = np.squeeze(pred, axis=0)
    #     # print(pred)
        
    #     predictions.append(pred)
    #     print(f"Prediction for prompt {i} has shape:", pred.shape)
        
    
    interfered_mse(haystack_prompts, model, device, config)
    
    # token_a = tokenizer.convert_tokens_to_ids("") # what it should predict
    # token_b = tokenizer.convert_tokens_to_ids("") # what it should predict if swap interferes properly (all encodings on the same token)
    
    # log_prob_a_changes, log_prob_b_changes = swap_token_activations_logits(
    #     model, haystack_prompts, args, token_a, token_b
    # )
    
    # plot_results(log_prob_a_changes, log_prob_b_changes, token_a, token_b)