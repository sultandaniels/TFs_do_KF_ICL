import torch
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial


#############################
##### Forward functions #####
#############################


def tokenize_for_ppl(model, tokenizer, prompts, completions):
    # Tokenize prompts and completions together
    prompt_completion_pairs = [
        prompt + completion for prompt, completion in zip(prompts, completions)
    ]
    tokenized_pairs = tokenizer(
        prompt_completion_pairs,
        padding=True,
        return_tensors="pt",
        max_length=model.config.max_position_embeddings,
        truncation=True,
    )

    # Prepare labels for calculating loss on completions only
    labels = tokenized_pairs["input_ids"].detach().clone()
    # Magic number 29901 is the colon (:) token for llama
    # Magic number 28747 is the colon (:) token for mistral
    completion_lengths = torch.argmax(
        (labels == 28747).to(torch.long).flip(dims=[1]), dim=1
    )
    labels = torch.cat(
        [
            labels[:, 1:],
            torch.tensor([[tokenizer.pad_token_id] for _ in range(len(labels))]).to(
                labels.device
            ),
        ],
        dim=1,
    )
    label_mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, length in enumerate(completion_lengths):
        label_mask[i, -length - 1 : -1] = True

    # Move tensors to the same device as the model
    input_ids = tokenized_pairs["input_ids"].to(model.device)
    attention_mask = tokenized_pairs["attention_mask"].to(model.device)
    labels = labels.to(model.device)
    label_mask = label_mask.to(model.device)
    return input_ids, attention_mask, labels, label_mask


def _forward(model, inputs, generate, no_grad, max_new_tokens=20, **kwargs):
    """
    Forward pass that bypasses tokenization.
    `inputs` should be a pre-prepared dict of inputs as expected by the model.
    """
    # Optionally preprocess inputs
    if "prepare_inputs" in kwargs:
        inputs = kwargs["prepare_inputs"](model, inputs)
        
    if no_grad:
        with torch.no_grad():
            if generate:
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    # If applicable, you may add a pad_token_id here
                    do_sample=False,
                    num_beams=1,
                )
            else:
                out = model.forward(**inputs)
    else:
        if generate:
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        else:
            out = model.forward(**inputs)
    return out



def _generate_single(model, tokenizer, tokenized, no_grad):
    if no_grad:
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_logits=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        out = model.generate(
            **tokenized,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out


######################################
##### Causal influence functions #####
######################################


def attn_head_influence(
    model,
    tokenizer,
    inputs,
    outputs,
    hooked_module,
    attn_head_activation=None,
    attn_head_idx=None,
):
    hook_handle = None
    input_ids, attention_mask, labels, label_mask = tokenize_for_ppl(
        model, tokenizer, inputs, outputs
    )

    def forward_pre_hook(module, input):
        if isinstance(input, tuple):
            new_input = input[0]
        else:
            new_input = input
        bsz, seq_len, _ = new_input.shape
        input_by_head = new_input.reshape(
            bsz, seq_len, model.config.num_attention_heads, -1
        )
        assert input_by_head.shape[-1] == attn_head_activation.shape[-1]
        prompt_idx = -torch.sum(label_mask, dim=1) - 1
        input_by_head[:, prompt_idx, attn_head_idx, :] = attn_head_activation.expand(
            bsz, -1
        )
        if isinstance(input, tuple):
            return (input_by_head.reshape(bsz, seq_len, -1),) + input[1:]
        return input_by_head.reshape(bsz, seq_len, -1)

    if attn_head_idx is not None and attn_head_activation is not None:
        hook_handle = hooked_module.register_forward_pre_hook(forward_pre_hook)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        perplexity_batch = torch.exp(
            (
                cross_entropy(logits.transpose(1, 2), labels, reduction="none")
                * label_mask
            ).sum(1)
            / label_mask.sum(1)
        )
    if hook_handle is not None:
        hook_handle.remove()
    return sum(perplexity_batch.tolist())


##############################
##### Generate functions #####
##############################


def generate_add_layer_n(
    model,
    tokenizer,
    inputs,
    hooked_module,
    layer_activation,
    token_idx=-1,
    no_grad=True,
    max_new_tokens=10,
):
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            new_output = output[0]
        else:
            new_output = output
        assert new_output.shape[-1] == layer_activation.shape[-1]
        new_output[:, token_idx, :] += layer_activation.expand(new_output.shape[0], -1)
        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        return new_output

    return_dict = {
        "clean_logits": [],
        "corrupted_logits": [],
        "corrupted_sequences": [],
    }
    tokenized = tokenizer(inputs, padding=True, return_tensors="pt").to(model.device)
    for _ in range(max_new_tokens):
        hook_handle = hooked_module.register_forward_hook(forward_hook)
        corrupted_out = _generate_single(model, tokenizer, tokenized, no_grad=no_grad)
        return_dict["corrupted_logits"].append(corrupted_out.logits[0])
        return_dict["corrupted_sequences"].append(corrupted_out.sequences[:, -1])
        hook_handle.remove()
        clean_out = _generate_single(model, tokenizer, tokenized, no_grad=no_grad)
        return_dict["clean_logits"].append(clean_out.logits[0])
        attn_mask = tokenized["attention_mask"]
        attn_mask = torch.cat(
            [attn_mask, attn_mask.new_ones((attn_mask.shape[0], 1))], dim=-1
        )
        tokenized = {
            "input_ids": clean_out.sequences,
            "attention_mask": attn_mask,
        }

    return_dict["corrupted_sequences"] = torch.stack(
        return_dict["corrupted_sequences"], dim=1
    )
    return_dict["clean_sequences"] = clean_out.sequences
    return return_dict


def generate_add_layer_single(
    model,
    tokenizer,
    inputs,
    hooked_module,
    layer_activation,
    token_idx=-1,
    no_grad=True,
    max_new_tokens=10,
):
    hook_triggered = [False]

    def forward_hook(module, input, output):
        if hook_triggered[0]:
            return None
        if isinstance(output, tuple):
            new_output = output[0]
        else:
            new_output = output
        assert new_output.shape[-1] == layer_activation.shape[-1]
        new_output[:, token_idx, :] += layer_activation.expand(new_output.shape[0], -1)
        hook_triggered[0] = True
        if isinstance(output, tuple):
            return (new_output,) + output[1:]
        return new_output

    hook_handle = hooked_module.register_forward_hook(forward_hook)
    out = _forward(
        model,
        tokenizer,
        inputs,
        generate=True,
        no_grad=no_grad,
        max_new_tokens=max_new_tokens,
    )
    hook_handle.remove()

    return out


def generate_substitute_layer_single_logits(
    model,
    inputs,
    hooked_modules,
    module_activations,
    sub_input_output,
    token_idx=0,
    no_grad=True,
    max_new_tokens=20,
    **kwargs,
):
    """
    Run a forward pass while substituting activations (for a single layer) and returns logits.
    This version assumes that `inputs` is already prepared and bypasses any tokenization.
    """
    assert len(hooked_modules) == len(module_activations)
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    hook_triggered = [False for _ in hooked_modules]

    def get_hook_by_idx(idx):
        def forward_pre_hook(module, input):
            if hook_triggered[idx]:
                return None
            new_input = input[0] if isinstance(input, tuple) else input
            if "substitute_by_mask" in kwargs:
                _, read_seq_len, _ = module_activations[idx].shape
                _, write_seq_len, _ = new_input.shape
                for i in range(len(new_input)):
                    read_mask = kwargs["substitute_by_mask"][i].item()
                    # If an attention mask is provided, use it; otherwise use the sequence length.
                    if "attention_mask" in inputs:
                        write_mask = inputs["attention_mask"][i].sum().item()
                    else:
                        write_mask = new_input.shape[1]
                    new_input[i] = torch.cat(
                        [
                            new_input[i][: write_seq_len - write_mask, :],
                            module_activations[idx][i, read_seq_len - read_mask :, :],
                            new_input[i][-(write_mask - read_mask) :, :],
                        ],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_input[:, token_idx, :].shape == new_activations.shape
                new_input[:, token_idx, :] = new_activations
            hook_triggered[idx] = True
            return (new_input,) + input[1:] if isinstance(input, tuple) else new_input

        def forward_hook(module, input, output):
            if hook_triggered[idx]:
                return None
            new_output = output[0] if isinstance(output, tuple) else output
            if "substitute_by_mask" in kwargs:
                _, read_seq_len, _ = module_activations[idx].shape
                _, write_seq_len, _ = new_output.shape
                for i in range(len(new_output)):
                    read_mask = kwargs["substitute_by_mask"][i].item()
                    if "attention_mask" in inputs:
                        write_mask = inputs["attention_mask"][i].sum().item()
                    else:
                        write_mask = new_output.shape[1]
                    new_output[i] = torch.cat(
                        [
                            new_output[i][: write_seq_len - write_mask, :],
                            module_activations[idx][i, read_seq_len - read_mask :, :],
                            new_output[i][-(write_mask - read_mask) :, :],
                        ],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_output[:, token_idx, :].shape == new_activations.shape
                new_output[:, token_idx, :] = new_activations

            hook_triggered[idx] = True
            return (new_output,) + output[1:] if isinstance(output, tuple) else new_output

        return forward_pre_hook if sub_input_output == "input" else forward_hook

    if sub_input_output == "input":
        hook_handles = [
            hooked_modules[i].register_forward_pre_hook(get_hook_by_idx(i))
            for i in range(len(hooked_modules))
        ]
    elif sub_input_output == "output":
        hook_handles = [
            hooked_modules[i].register_forward_hook(get_hook_by_idx(i))
            for i in range(len(hooked_modules))
        ]
    else:
        raise ValueError("sub_input_output must be 'input' or 'output'.")

    # Ensure inputs are not raw strings.
    if isinstance(inputs, str):
        raise ValueError("Raw string input provided but tokenization is disabled. Provide pre-processed inputs instead.")
    
    with (torch.no_grad() if no_grad else torch.enable_grad()):
        # Call the updated _forward function (which no longer takes a tokenizer)
        out = _forward(model, inputs, generate=False, no_grad=no_grad, **kwargs)
        logits = out.logits

    for hook_handle in hook_handles:
        hook_handle.remove()

    return logits


def generate_substitute_layer_single(
    model,
    inputs,
    hooked_modules,
    module_activations,
    sub_input_output,
    token_idx=0,
    no_grad=True,
    max_new_tokens=20,
    **kwargs,
):
    """
    Run a forward pass with substituted activations (for a single layer) and return the model output.
    This version bypasses tokenization, assuming that inputs are already formatted appropriately.
    """
    assert len(hooked_modules) == len(module_activations)
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    hook_triggered = [False for _ in hooked_modules]

    def get_hook_by_idx(idx):
        def forward_pre_hook(module, input):
            if hook_triggered[idx]:
                return None
            new_input = input[0] if isinstance(input, tuple) else input
            if "substitute_by_mask" in kwargs:
                _, read_seq_len, _ = module_activations[idx].shape
                _, write_seq_len, _ = new_input.shape
                for i in range(len(new_input)):
                    read_mask = kwargs["substitute_by_mask"][i].item()
                    if "attention_mask" in inputs:
                        write_mask = inputs["attention_mask"][i].sum().item()
                    else:
                        write_mask = new_input.shape[1]
                    new_input[i] = torch.cat(
                        [
                            new_input[i][: write_seq_len - write_mask, :],
                            module_activations[idx][i, read_seq_len - read_mask :, :],
                            new_input[i][-(write_mask - read_mask) :, :],
                        ],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_input[:, token_idx, :].shape == new_activations.shape
                new_input[:, token_idx, :] = new_activations
            hook_triggered[idx] = True
            return (new_input,) + input[1:] if isinstance(input, tuple) else new_input

        def forward_hook(module, input, output):
            if hook_triggered[idx]:
                return None
            new_output = output[0] if isinstance(output, tuple) else output
            if "substitute_by_mask" in kwargs:
                _, read_seq_len, _ = module_activations[idx].shape
                _, write_seq_len, _ = new_output.shape
                for i in range(len(new_output)):
                    read_mask = kwargs["substitute_by_mask"][i].item()
                    if "attention_mask" in inputs:
                        write_mask = inputs["attention_mask"][i].sum().item()
                    else:
                        write_mask = new_output.shape[1]
                    new_output[i] = torch.cat(
                        [
                            new_output[i][: write_seq_len - write_mask, :],
                            module_activations[idx][i, read_seq_len - read_mask :, :],
                            new_output[i][-(write_mask - read_mask) :, :],
                        ],
                        dim=0,
                    )
            else:
                new_activations = module_activations[idx].expand(-1, len(token_idx), -1)
                assert new_output[:, token_idx, :].shape == new_activations.shape
                new_output[:, token_idx, :] = new_activations

            hook_triggered[idx] = True
            return (new_output,) + output[1:] if isinstance(output, tuple) else new_output

        return forward_pre_hook if sub_input_output == "input" else forward_hook

    if sub_input_output == "input":
        hook_handles = [
            hooked_modules[i].register_forward_pre_hook(get_hook_by_idx(i))
            for i in range(len(hooked_modules))
        ]
    elif sub_input_output == "output":
        hook_handles = [
            hooked_modules[i].register_forward_hook(get_hook_by_idx(i))
            for i in range(len(hooked_modules))
        ]
    else:
        raise ValueError("sub_input_output must be 'input' or 'output'.")

    # Decide whether to generate or calculate loss.
    if "get_loss" in kwargs:
        if "labels" not in inputs:
            raise ValueError("Expected 'labels' in inputs when 'get_loss' is specified.")
        out = _forward(model, inputs, generate=False, no_grad=(not kwargs["get_loss"]), **kwargs)
    else:
        out = _forward(model, inputs, generate=True, no_grad=no_grad, max_new_tokens=max_new_tokens, **kwargs)

    for hook_handle in hook_handles:
        hook_handle.remove()

    return out



def generate_add_attn_single(
    model,
    tokenizer,
    inputs,
    hooked_module,
    attn_head_idx,
    attn_head_activation,
    token_idx=-1,
    no_grad=True,
):
    hook_triggered = [False]

    def forward_pre_hook(module, input):
        if hook_triggered[0]:
            return None
        if isinstance(input, tuple):
            new_input = input[0]
        else:
            new_input = input
        bsz, seq_len, _ = new_input.shape
        input_by_head = new_input.reshape(
            bsz, seq_len, model.config.num_attention_heads, -1
        )
        input_by_head[:, token_idx, attn_head_idx, :] += attn_head_activation.expand(
            bsz, -1
        )
        hook_triggered[0] = True
        if isinstance(input, tuple):
            return (input_by_head.reshape(bsz, seq_len, -1),) + input[1:]
        return input_by_head.reshape(bsz, seq_len, -1)

    hook_handle = hooked_module.register_forward_pre_hook(forward_pre_hook)
    out = _forward(model, tokenizer, inputs, generate=True, no_grad=no_grad)
    hook_handle.remove()

    return out


#############################
##### Caching functions #####
#############################


def _forward_cache_outputs(model, inputs, hooked_modules, token_idx, no_grad=True, **kwargs):
    """
    Executes a forward pass while caching the outputs of the given hooked modules.
    """
    cache = []

    def forward_hook(module, input, output):
        # If output is a tuple, grab the first element.
        if isinstance(output, tuple):
            output = output[0]
        # If token_idx is specified, grab that slice; otherwise, cache full output.
        if token_idx is None:
            cache.append(output)
        else:
            cache.append(output[:, token_idx, :])
        return None

    hook_handles = [
        hooked_module.register_forward_hook(forward_hook)
        for hooked_module in hooked_modules
    ]
    _ = _forward(model, inputs, generate=False, no_grad=no_grad, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return cache


def _forward_cache_inputs(
    model, tokenizer, inputs, hooked_modules, split, token_idx, no_grad=True, **kwargs
):
    cache = []

    def forward_pre_hook_idx(idx):
        def forward_pre_hook(module, input):
            if isinstance(input, tuple):
                input = input[0]
            if split[idx]:
                bsz, seq_len, _ = input.shape
                input_by_head = input.reshape(
                    bsz, seq_len, model.config.num_attention_heads, -1
                )
                if token_idx is None:
                    cache.append(input_by_head)
                else:
                    cache.append(input_by_head[:, token_idx, :, :])
            else:
                if token_idx is None:
                    cache.append(input)
                else:
                    cache.append(input[:, token_idx, :])
            return None

        return forward_pre_hook

    hook_handles = [
        hooked_modules[i].register_forward_pre_hook(forward_pre_hook_idx(i))
        for i in range(len(hooked_modules))
    ]
    _ = _forward(model, tokenizer, inputs, generate=False, no_grad=no_grad, **kwargs)
    for hook_handle in hook_handles:
        hook_handle.remove()
    return cache


def cache_activations(
    model,
    module_list_or_str,
    cache_input_output,
    inputs,
    batch_size,
    token_idx=-1,
    split_attn_by_head=True,
    **kwargs,
):
    # Ensure token_idx is a list for consistency.
    if isinstance(token_idx, int):
        token_idx = [token_idx]
    # Allow module specification via a string or a list of strings.
    if isinstance(module_list_or_str, str):
        module_strs = [module_list_or_str]
    else:
        module_strs = module_list_or_str
        
    if split_attn_by_head and cache_input_output == "input":
        split = [True if "attn" in m else False for m in module_strs]
    else:
        split = [False for _ in module_strs]

    all_activations = [None for _ in module_strs]
    modules = [eval(m) for m in module_strs]

    # Process inputs in batches
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]
        if cache_input_output == "input":
            activations = _forward_cache_inputs(model, batch, modules, split, token_idx, **kwargs)
        elif cache_input_output == "output":
            activations = _forward_cache_outputs(model, batch, modules, token_idx, **kwargs)
        else:
            raise ValueError("cache_input_output must be 'input' or 'output'")
        # Concatenate activations batch by batch.
        for j, activation in enumerate(activations):
            if i == 0:
                all_activations[j] = activation
            else:
                all_activations[j] = torch.cat([all_activations[j], activation], dim=0)
    return all_activations



##############################
##### Model loading code #####
##############################



def load_model(args):
    model_name_or_path = args.model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    if any(
        [
            n in model_name_or_path
            for n in ["llama", "zephyr", "gemma", "mistral", "Qwen"]
        ]
    ):
        module_str_dict = {
            "layer": "model.model.layers[{layer_idx}]",
            "attn": "model.model.layers[{layer_idx}].self_attn.o_proj",
        }
        n_layers = len(model.model.layers)
    elif "gpt-j" or "GPT" in model_name_or_path:
        module_str_dict = {
            "layer": "model.transformer.h[{layer_idx}]",
            "attn": "model.transformer.h[{layer_idx}].attn.o_proj",
        }
        n_layers = len(model.transformer.h)
    elif "opt" in model_name_or_path:
        module_str_dict = {
            "layer": "model.model.decoder.layers[{layer_idx}]",
            "attn": "model.model.decoder.layers[{layer_idx}].self_attn.o_proj",
        }
        n_layers = len(model.model.decoder.layers)
    args.module_str_dict = module_str_dict
    args.n_layers = n_layers
    return model, tokenizer

def get_modules(model):
    modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            modules.append(name)
    return modules


def print_hooks(module):
    no_hooks = False
    # Print forward hooks
    if hasattr(module, "_forward_hooks") and len(module._forward_hooks) > 0:
        no_hooks = True
        print(f"{module.__class__.__name__} has the following forward hooks:")
        for hook_id, hook in module._forward_hooks.items():
            print(f"\tHook ID: {hook_id}, Hook: {hook}")

    # Print backward hooks
    if hasattr(module, "_backward_hooks") and len(module._backward_hooks) > 0:
        no_hooks = True
        print(f"{module.__class__.__name__} has the following backward hooks:")
        for hook_id, hook in module._backward_hooks.items():
            print(f"\tHook ID: {hook_id}, Hook: {hook}")

    # Print forward pre-hooks
    if hasattr(module, "_forward_pre_hooks") and len(module._forward_pre_hooks) > 0:
        no_hooks = True
        print(f"{module.__class__.__name__} has the following forward pre-hooks:")
        for hook_id, hook in module._forward_pre_hooks.items():
            print(f"\tHook ID: {hook_id}, Hook: {hook}")

    if no_hooks:
        print(f"{module.__class__.__name__} has no hooks.")
