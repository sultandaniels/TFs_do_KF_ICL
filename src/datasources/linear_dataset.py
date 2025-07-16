import pickle
import numpy as np
from datasources.filter_dataset import generate_zipfian_integer, special_tokens
import torch
from core import Config

config = Config()

def generate_seg_lens(n_positions, sys_in_trace):
    rng = np.random.default_rng()

    # generate a sample from a poisson dist and name it num_cut
    lam = 2*sys_in_trace

    num_cut = rng.poisson(lam) # number of cuts in the trace

    rel_positions = rng.integers(0, int(n_positions/2), size=num_cut)
    positions = rel_positions*2
    if not 0 in positions:
        positions = np.append(positions, 0)
    positions = np.append(positions, n_positions)
    positions.sort() # sort the positions in ascending order

    diffs = np.diff(positions)
    return diffs - 2


def populate_traces(config, num_tasks, entries):
    sys_choices = [] #list that will hold the order of the system choices for the trace
    seg_starts = []
    tok_seg_lens = []
    real_seg_lens = []

    context_len = config.n_positions + 1 #the length of the context is the number of positions plus 1 for the start token

    sys_names = np.arange(config.max_sys_trace) #system names
    #randomly shuffle the system names to assign to the system indices for the open and close tokens
    np.random.shuffle(sys_names)

    sys_in_trace = generate_zipfian_integer(config.max_sys_trace, 1.5) #number of systems to include in the context
    rng = np.random.default_rng()
    sys_inds = rng.choice(num_tasks, sys_in_trace, replace=False).tolist()

    sys_dict = {}
    for i in range(len(sys_inds)):
        sys_dict[sys_inds[i]] = sys_names[i]

    seg_lens = generate_seg_lens((context_len - 1), sys_in_trace)

    segments = np.zeros((context_len, config.nx + config.ny + 2*config.max_sys_trace + 3))
    segments[0, 2*config.max_sys_trace] = np.sqrt(2) #set the start token for the first segment

    next_start = {sys_ind: 0 for sys_ind in sys_inds}

    seg_start = 1
    seg_count = 0
    for seg_len in seg_lens:
        seg_starts.append(seg_start)
        sys_ind = np.random.choice(sys_inds)
        sys_choices.append(sys_ind)

        sys_trace_obs_x = entries[sys_ind]["x"]
        sys_trace_obs_y = entries[sys_ind]["y"]

        if seg_len == -2: #two closed parens on top of each other
            tok_seg_lens.append(0)
            real_seg_lens.append(0)
            seg_count += 1
            continue

        elif seg_len == 0: #closed paren, open paren, closed paren
            start_paren, end_paren = special_tokens(segments, sys_dict[sys_ind], style="zeros") #get the special tokens for the segment
            tok_seg_len = 2
            tok_seg_lens.append(tok_seg_len)
            real_seg_lens.append(0)

            try:
                segments[seg_start:seg_start + tok_seg_len, :] = np.concatenate([start_paren, end_paren], axis=0) #open paren, closed paren
            except ValueError as e:
                print(f"seg_start: {seg_start}, tok_seg_len: {tok_seg_len}, context_len: {context_len}")
                print(f"segments[seg_start:seg_start + tok_seg_len, :].shape: {segments[seg_start:seg_start + tok_seg_len, :].shape}")
                print(f"start_paren.shape: {start_paren.shape}, end_paren.shape: {end_paren.shape}")
                print(f"seg_starts: {seg_starts}")
                raise ValueError(e)

            if seg_start + tok_seg_len == context_len:
                break

            seg_start += tok_seg_len #update the starting index for the next segment
            seg_count += 1
            continue
        else:
            if next_start[sys_ind] + int(seg_len/2) > sys_trace_obs_x.shape[0]: #if the next starting index plus the segment length is greater than the length of the trace
                if next_start[sys_ind] >= sys_trace_obs_x.shape[0]: #if the next starting index is greater than the length of the trace, skip to the next trace
                    continue
                else:
                    segment_x = sys_trace_obs_x[next_start[sys_ind]:, :] #get the segment from the next starting index to the end of the trace
                    segment_y = sys_trace_obs_y[next_start[sys_ind]:, :] #get the segment from the next starting index to the end of the trace
                    seg_len = 2*segment_x.shape[0]
            else:
                segment_x = sys_trace_obs_x[next_start[sys_ind]:next_start[sys_ind] + int(seg_len/2), :] #get the segment from the next starting index to the next starting index plus the segment length
                segment_y = sys_trace_obs_y[next_start[sys_ind]:next_start[sys_ind] + int(seg_len/2), :] #get the segment from the next starting index to the next starting index plus the segment length

            # concatenate 1 columns of ones to the segment
            ones = np.ones((segment_x.shape[0], 1))
            segment_x = np.concatenate((ones, segment_x), axis=1)
            segment_y = np.concatenate((ones, segment_y), axis=1)
        
            zeros_x = np.zeros((segment_x.shape[0], segment_y.shape[1]))
            zeros_y = np.zeros((segment_y.shape[0], segment_x.shape[1]))
            segment_x = np.concatenate((segment_x, zeros_x), axis=1)
            segment_y = np.concatenate((zeros_y, segment_y), axis=1)

            segment = np.zeros((2*segment_x.shape[0], segment_x.shape[1]))
            for i in range(segment_x.shape[0]):
                segment[2*i, :] = segment_x[i, :]
                segment[2*i + 1, :] = segment_y[i, :]

            zeros = np.zeros((segment.shape[0], 2*config.max_sys_trace + 1))
            segment = np.concatenate((zeros, segment), axis=1)
        
            start_paren, end_paren = special_tokens(segment, sys_dict[sys_ind], style="zeros") #get the special tokens for the segment

            segment = np.concatenate([start_paren, segment, end_paren], axis=0) #concatenate the special tokens to the segment

            if seg_start + seg_len + 2 > context_len:
                #truncate the segment if it is too long so that it fits in the context
                segment = segment[:context_len - seg_start, :]
                seg_len = segment.shape[0] - 1

            tok_seg_len = segment.shape[0]
            tok_seg_lens.append(tok_seg_len)
            real_seg_lens.append(seg_len)

            segments[seg_start:seg_start + tok_seg_len, :] = segment #add the segment to the segments array

            next_start[sys_ind] += int(seg_len/2) #update the next starting index for the trace from this system index 

            if seg_start + tok_seg_len == context_len:
                break

            seg_start += tok_seg_len #update the starting index for the next segment
            seg_count += 1

    return segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds


class LinearDataset:
    def __init__(self, path, use_true_len=config.use_true_len):
        super(LinearDataset, self).__init__()
        self.load(path)
        self.use_true_len = use_true_len

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        self.data = data

    def __len__(self):
        if not self.use_true_len:
            return config.train_steps * config.batch_size #have the dataset length be the # of training steps
        else: 
            return len(self.data) #have the dataset length be the number of training traces
        

    def __getitem__(self, idx):
        segments, sys_choices, sys_dict, seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(config, config.num_tasks, self.data)
        entry = {"current": segments[:-1, :], "target": segments[1:, config.nx + 2*config.max_sys_trace + 3:]} #create the entry dictionary with the current and target segments, where the target segment has only the config.ny columns
        torch_entry = dict([
                (k, (torch.from_numpy(a) if isinstance(a, np.ndarray) else a).to(torch.float32))
                for k, a in entry.items()])
        return torch_entry