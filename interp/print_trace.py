import numpy as np
import pickle

filename = "/data/dhruv_gautam/TFs_do_KF_ICL/outputs/GPT2/250112_043028.07172b_multi_sys_trace_ortho_state_dim_5_ident_C_lr_1.584893192461114e-05_num_train_sys_40000/data/interleaved_traces_ortho_ident_C_state_dim_5_num_sys_haystack_1.pkl"

with open(filename, 'rb') as f:
    file_dict = pickle.load(f)

multi_sys_ys = file_dict["multi_sys_ys"]
tok_seg_lens_per_config = file_dict["tok_seg_lens_per_config"]
sys_choices_per_config = file_dict["sys_choices_per_config"]

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

print("Length of first segment (token dimension):", len(multi_sys_ys[0, 0, 0, :]))  # e.g., 57 tokens
print("Number of segments in one trace:", len(multi_sys_ys[0, 0, :, :]))         # e.g., 25 segments

print(len(sys_choices_per_config))
for config_idx, sys_choices in enumerate(sys_choices_per_config):
    print(f"\nConfiguration {config_idx}:")
    
    # The number of interleaved segments is given by the haystack (25 in this case).
    num_interleaved_segments = len(tok_seg_lens_per_config[config_idx])
    num_orig_segs = len(sys_choices)
    print(num_orig_segs)
    
    # If the original system choices list length doesn't match 25, we assume the interleaving
    # is done cyclically (round-robin across the available systems).
    for seg_idx in range(num_interleaved_segments):
        # Using modulo to map the haystack segment index to the original system index.
        system_for_segment = sys_choices[seg_idx % num_orig_segs]
        
        # Optionally, you can also look up the token length for this segment in a similar way.
        token_length = tok_seg_lens_per_config[config_idx][seg_idx % num_orig_segs]
        
        print(f"  Interleaved Segment {seg_idx}: System = {system_for_segment}, seq length = {token_length}")
    
    # Print unique systems present in this interleaved trace.
    interleaved_systems = {sys_choices[seg_idx % num_orig_segs] for seg_idx in range(num_interleaved_segments)}
    if len(interleaved_systems) > 1:
        print("-> This trace is interleaved with systems:", interleaved_systems)
    else:
        # There’s only one unique system.
        print("-> This trace is not interleaved; it contains only system:", interleaved_systems.pop())