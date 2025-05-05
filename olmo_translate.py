#!/usr/bin/env python
import json
import re
import argparse
import torch
from hf_olmo import OLMoForCausalLM
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_examples(path):
    """
    Load your JSON with separate 'input' and 'target' fields,
    then concatenate each into one unified string.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    task_prefix = cfg["task_prefix"]
    raw = cfg["examples"]
    unified = [
        ex["input"].strip() + " " + ex["target"].strip()
        for ex in raw
    ]
    return task_prefix, unified

def list_step_revisions(model_id):
    api = HfApi()
    refs = api.list_repo_refs(repo_id=model_id).branches
    steps = [r.name for r in refs if r.name.startswith("step")]
    steps.sort(key=lambda name: int(name.split("step")[1].split("-")[0]))
    return steps

def predict_token_id(prompt: str, model, tokenizer, device="cuda") -> int:
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **tokens,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # the new token ID is at position len(input_ids)
    return gen[0, tokens["input_ids"].shape[-1]].item()

def plot_metric(df, metric_name):
    df['step'] = df['revision'].apply(lambda r: int(r.split('step')[1].split('-')[0]))
    df = df.sort_values('step')
    plt.figure()
    plt.plot(df['step'], df[metric_name], marker='o')
    plt.xlabel('Training step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs. training step')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OLMo-7B first/second token accuracy"
    )
    parser.add_argument("--examples", "-e", required=True,
                        help="JSON with `task_prefix` and examples with 'input'/'target'")
    parser.add_argument("--model", "-m", default="allenai/OLMo-7B",
                        help="HuggingFace Hub model ID")
    parser.add_argument("--output", "-o", default="token_acc_results.csv",
                        help="Where to save CSV results")
    parser.add_argument("--device", "-d", default="cuda",
                        help="Torch device (e.g. cuda or cpu)")
    args = parser.parse_args()

    task_prefix, examples = load_examples(args.examples)
    revisions = list_step_revisions(args.model)
    print(f"Found {len(revisions)} revisions:", revisions)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results = []

    for rev in tqdm(revisions, desc="Revisions"):
        model = OLMoForCausalLM.from_pretrained(
            args.model, revision=rev
        ).to(args.device)
        model.eval()

        first_correct = second_correct = 0
        count1 = count2 = 0

        for uni in examples:
            # find the second occurrence of "IPA:" or "English:"
            occ = list(re.finditer(r"(?:IPA:|English:)", uni))
            if len(occ) < 2:
                continue
            split_idx = occ[1].end()

            # build the prompt up to that point
            prompt_base = task_prefix + uni[:split_idx].rstrip() + " "
            target_text = uni[split_idx:]

            # tokenize the true continuation
            true_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
            if not true_ids:
                continue

            # --- FIRST token (skip pure‐whitespace tokens) ---
            t1 = true_ids[0]
            s1 = tokenizer.decode([t1])
            if not s1.isspace():
                p1 = predict_token_id(prompt_base, model, tokenizer, device=args.device)
                if p1 == t1:
                    first_correct += 1
                count1 += 1

            # --- SECOND token, conditioning on GT first ---
            if len(true_ids) >= 2:
                t2 = true_ids[1]
                s2 = tokenizer.decode([t2])
                if not (s1.isspace() or s2.isspace()):
                    prompt2 = prompt_base + s1
                    p2 = predict_token_id(prompt2, model, tokenizer, device=args.device)
                    if p2 == t2:
                        second_correct += 1
                    count2 += 1

        # compute accuracies
        acc1 = first_correct  / count1  if count1  else 0.0
        acc2 = second_correct / count2 if count2 else 0.0

        results.append({
            "revision": rev,
            "first_token_acc":  acc1,
            "second_token_acc": acc2
        })

        # free GPU
        del model
        torch.cuda.empty_cache()

    # Save and plot
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"✅ Results written to {args.output}")

    for m in ["first_token_acc", "second_token_acc"]:
        plot_metric(df, m)

if __name__ == "__main__":
    main()
