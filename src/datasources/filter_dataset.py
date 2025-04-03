from torch.utils.data import Dataset
from dyn_models.filtering_lti import *
from core import Config
import torch
import scipy.stats as stats
import pickle
from linalg_helpers import print_matrix


config = Config()

def generate_zipfian_integer(n, a):
    """
    Generate integer number between 1 and n (inclusive) from a Zipf's power law distribution.

    Parameters:
    n (int): The upper limit (inclusive) for the range of integers.
    a (float): The parameter of the Zipfian distribution (a >= 0).

    Returns:
    int: An integer between 0 and n.
    """
    # Generate samples from a Zipfian distribution
    sample = stats.zipfian.rvs(a,n, size=1)

    return sample[0]

def generate_seg_lens(n_positions, sys_in_trace):
    """
    Generate segment lengths for a trace.

    Parameters:
    config.n_positions (int): The number of positions in the trace.
    sys_in_trace (int): The number of systems in the trace.

    # create a random generator
    """
    rng = np.random.default_rng()

    # generate a sample from a poisson dist and name it num_cut
    lam = 2*sys_in_trace

    num_cut = rng.poisson(lam) # number of cuts in the trace

    # generate num_cut random cut positions in trace distributed uniformly over the entire context
    positions = rng.integers(0, n_positions, size=num_cut) #positions are the index of the closed paren (and start token)
    if not 0 in positions:
        positions = np.append(positions, 0)
    positions = np.append(positions, n_positions)
    positions.sort() # sort the positions in ascending order

    # calculate the difference between successive positions
    diffs = np.diff(positions)

    # calculate the real segment lengths (just the number of observations in each segment)
    seg_lens = diffs - 2
    return seg_lens

def print_matrix(matrix, name):
    """
    Print a matrix in a readable format.
    
    Parameters:
    matrix (np.ndarray): The matrix to print.
    name (str): The name of the matrix.
    """
    print(f"Matrix {name}:")
    rows, cols = matrix.shape #get the number of rows and columns in the matrix
    for i in range(rows):
        for j in range(cols):
            print(f"{matrix[i, j]:>10.4f}", end=" ") #print the element in the matrix
        print()

def special_tokens(segment, sys_name, style):
    # Create the special tokens
    if style == "big_num":
        start_token = (100 * (sys_name + 1)) * np.ones((1, segment.shape[1]))
        end_token = (100 * (sys_name + 1) + 1) * np.ones((1, segment.shape[1]))
    elif style == "frac":
        start_token = (sys_name/(sys_name + 1)) * np.ones((1, segment.shape[1]))
        end_token = (-sys_name/(sys_name + 1)) * np.ones((1, segment.shape[1]))
    elif style == "zeros":
        # create an array of zeros with the same number of columns as the segment, but with a sqrt2 in the column corresponding to the 2*system name
        start_token = np.zeros((1, segment.shape[1]))
        start_token[0, 2*sys_name] = np.sqrt(2)
        end_token = np.zeros((1, segment.shape[1]))
        end_token[0, 2*sys_name + 1] = np.sqrt(2)
    else:
        raise ValueError(f"Special token style {style} has not been implemented.")
    
    return start_token, end_token

def populate_traces(config, num_tasks, entries, test=False, train_conv=False, trace_conf=None, example=None):

    sys_choices = [] #list that will hold the order of the system choices for the trace
    seg_starts = []
    tok_seg_lens = []
    real_seg_lens = []

    context_len = config.n_positions + 1 #the length of the context is the number of positions plus 1 for the start token


    sys_names = np.arange(config.max_sys_trace) #system names
    #randomly shuffle the system names to assign to the system indices for the open and close tokens
    np.random.shuffle(sys_names)
    
    if config.single_system: #if single sys multi segment test (DEPRECATED)
        sys_in_trace = 1
        sys_inds = [0] #list of system indices
        sys_dict = {sys_inds[0]: sys_names[0]}

        if train_conv:
            seg_lens = [int(config.n_positions/2 - 2)]*2
        else:
            seg_lens = generate_seg_lens(config.n_positions, sys_in_trace)

    else:

        #Set the number of systems to include in the context
        if config.needle_in_haystack:
            sys_in_trace = config.num_sys_haystack #number of systems to include in the context
        elif config.zero_cut:
            sys_in_trace = 1
        else:
            sys_in_trace = generate_zipfian_integer(config.max_sys_trace, 1.5) #number of systems to include in the context

        #set the system indices to be used from the library of traces
        if config.zero_cut and test: #zero cut test
            sys_inds = [trace_conf] #set the system index to the trace_conf
        elif config.needle_in_haystack and test:
            if config.fix_needle and config.num_sys_haystack == 2:
                start_sys = 0
                sys_inds = [start_sys, start_sys + example + 1] #Fix the needle to be the same system for each example
                # sys_inds = [start_sys + example + 1, start_sys] #Fix the other system to be the same system for each example
            else:
                sys_inds = np.arange(example, example + sys_in_trace) #set the system indices for the specific example
        else:
            #uniformly at random select sys_in_traces numbers between 0 and num_tasks without replacement for the system indices
            rng = np.random.default_rng()
            sys_inds = rng.choice(num_tasks, sys_in_trace, replace=False).tolist()

        #create a tuple that matches the system names to the system indices
        sys_dict = {}
        if config.irrelevant_tokens:
            for i in range(len(sys_inds) + 1):
                if i < len(sys_inds):
                    sys_dict[sys_inds[i]] = sys_names[i]
                else:
                    sys_dict[sys_inds[-1] + 1] = sys_names[i] #add the irrelevant token to the dictionary

        else:
            for i in range(len(sys_inds)):
                if config.same_tokens: #use the same tokens for all systems
                    sys_dict[sys_inds[i]] = sys_names[0]
                else:
                    sys_dict[sys_inds[i]] = sys_names[i]
        

        #set the segment lengths for the context
        if config.needle_in_haystack: #needle in haystack test
            if config.needle_final_seg_extended:
                seg_lens = [config.len_seg_haystack]*(config.num_sys_haystack - 1) + [context_len - ((config.num_sys_haystack - 1)*(config.len_seg_haystack + 2) + 2)]
                print(f"final seg extended seg lens: {seg_lens}")
            else:
                seg_lens = [config.len_seg_haystack]*config.num_sys_haystack + [context_len - (1 + config.num_sys_haystack*(config.len_seg_haystack + 2) + 2)]

        else: #train or non needle in haystack test
            if config.zero_cut: #zero cut
                seg_lens = [config.n_positions - 2]
            else:
                seg_lens = generate_seg_lens(config.n_positions, sys_in_trace)

    segments = np.zeros((context_len, config.ny + 2*config.max_sys_trace + 2)) #initialize the segments array
    segments[0, 2*config.max_sys_trace] = np.sqrt(2) #set the start token for the first segment

    #initialize a dictionary to hold the next starting index for each system trace
    if config.late_start is not None:
        next_start_ind = config.late_start
    else:
        next_start_ind = 0

    next_start = {sys_ind: next_start_ind for sys_ind in sys_inds} 

    seg_start = 1 #initialize the starting index for the segment at 1 to account for the start token

    seg_count = 0
    for seg_len in seg_lens:

        seg_starts.append(seg_start)

        if config.single_system: #if single sys multi segment test
            sys_ind = sys_inds[0]
            sys_choices.append(sys_ind) #add the system index to the list of system choices

        else:

            if config.needle_in_haystack:
                #pick the system index from the list of system indices
                ind = seg_count % len(sys_inds) #use mod to cycle through the system indices
                sys_ind = sys_inds[ind]
            else:
                #pick a random system index
                sys_ind = np.random.choice(sys_inds)
                if config.zero_cut and test:
                    print(f"sys_ind: {sys_ind}")

            sys_choices.append(sys_ind) #add the system index to the list of system choices

        #get obs from the system trace corresponding to sys_trace_ind
        if test:
            #catch exception if the system index is not in the entries
            try:
                sys_trace_obs = entries[sys_ind]
            except IndexError as e:
                print(f"System index {sys_ind} is out of bounds from sys_inds: {sys_inds}")
                raise IndexError(e)
            

        else:
            sys_trace_obs = entries[sys_ind]["obs"]

        if seg_len == -2: #two closed parens on top of each other
            
            tok_seg_lens.append(0)
            real_seg_lens.append(0)
            seg_count += 1
            continue

        elif seg_len == -1: # #two closed parens one after the other

            start_paren, end_paren = special_tokens(segments, sys_dict[sys_ind], style="zeros") #get the special tokens for the segment
            tok_seg_len = 1 
            tok_seg_lens.append(tok_seg_len)
            real_seg_lens.append(0) 

            segments[seg_start:seg_start + tok_seg_len, :] = end_paren #closed paren

            if seg_start + tok_seg_len == context_len:
                break

            seg_start += tok_seg_len #update the starting index for the next segment
            seg_count += 1
            continue
        elif seg_len == 0: #closed paren, open paren, closed paren

            start_paren, end_paren = special_tokens(segments, sys_dict[sys_ind], style="zeros") #get the special tokens for the segment
            tok_seg_len = 2
            tok_seg_lens.append(tok_seg_len)
            real_seg_lens.append(0)

            segments[seg_start:seg_start + tok_seg_len, :] = np.concatenate([start_paren, end_paren], axis=0) #open paren, closed paren

            if seg_start + tok_seg_len == context_len:
                break

            seg_start += tok_seg_len #update the starting index for the next segment
            seg_count += 1
            continue
        else:

            if next_start[sys_ind] + seg_len > sys_trace_obs.shape[0]: #if the next starting index plus the segment length is greater than the length of the trace
                if next_start[sys_ind] >= sys_trace_obs.shape[0]: #if the next starting index is greater than the length of the trace, skip to the next trace
                    continue
                else:
                    segment = sys_trace_obs[next_start[sys_ind]:, :] #get the segment from the next starting index to the end of the trace
                    seg_len = segment.shape[0] #update the segment length to the length of the segment
            else:
                segment = sys_trace_obs[next_start[sys_ind]:next_start[sys_ind] + seg_len, :] #get the segment from the next starting index to the next starting index plus the segment length

            # concatenate 1 columns of ones to the segment
            ones = np.ones((segment.shape[0], 1))
            segment = np.concatenate((ones, segment), axis=1)
        
            # concatenate 2*config.max_sys_trace + 1 columns of zeros to the segment
            zeros = np.zeros((segment.shape[0], 2*config.max_sys_trace + 1))
            segment = np.concatenate((zeros, segment), axis=1)

            if test and config.needle_in_haystack and config.paren_swap and len(seg_starts) == config.num_sys_haystack + 1: #swap open token for query experiment

                swap_sys_ind = sys_inds[int((seg_count + 1) % config.num_sys_haystack)] #swap the system index for the query to the next system index in a cycle
                # print("sys_dict: ", sys_dict) 
                # print(f"sys_ind: {sys_ind}, sys_dict[sys_ind]: {sys_dict[sys_ind]}, swap_sys_ind: {swap_sys_ind}, sys_dict[swap_sys_ind]: {sys_dict[swap_sys_ind]}")

                start_paren, end_paren = special_tokens(segment, sys_dict[swap_sys_ind], style="zeros") #get the special tokens for the segment

            elif test and config.needle_in_haystack and config.irrelevant_tokens and len(seg_starts) == config.num_sys_haystack + 1: #give irrelevant open token for query experiment

                irr_sys_ind = sys_inds[-1] + 1

                # print("in populate_traces")

                # print("sys_dict: ", sys_dict) 
                # print(f"sys_ind: {sys_ind}, sys_dict[sys_ind]: {sys_dict[sys_ind]}, irr_sys_ind: {irr_sys_ind}, sys_dict[irr_sys_ind]: {sys_dict[irr_sys_ind]}\n\n")

                start_paren, end_paren = special_tokens(segment, sys_dict[irr_sys_ind], style="zeros") #get the special tokens for the segment

            else:
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

            next_start[sys_ind] += seg_len #update the next starting index for the trace from this system index 

            if seg_start + tok_seg_len == context_len:
                break

            seg_start += tok_seg_len #update the starting index for the next segment
            seg_count += 1
        
    return segments, sys_choices, sys_dict, tok_seg_lens, seg_starts, real_seg_lens, sys_inds


class FilterDataset(Dataset):
    def __init__(self, path, use_true_len=config.use_true_len):
        super(FilterDataset, self).__init__()
        self.load(path)
        self.use_true_len = use_true_len

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            entries = []

            if len(data[0].keys()) > 1:
                key_to_extract = 'obs'
                # training dataset is only the observations
                for item in data:
                    if key_to_extract in item:
                        entries.append({key_to_extract: item[key_to_extract]})
            else:
                entries = data

            self.entries = entries


    def __len__(self): 
        if not self.use_true_len:
            return config.train_steps * config.batch_size #have the dataset length be the # of training steps
        else: 
            return len(self.entries) #have the dataset length be the number of training traces
        

    def __getitem__(self, idx):

        #Currently the algorithm can choose the same system twice in a row
        if config.multi_sys_trace:
            segments, sys_choices, sys_dict, seg_lens, seg_starts, real_seg_lens, sys_inds = populate_traces(config, config.num_tasks, self.entries)
            entry = {"current": segments[:-1, :], "target": segments[1:, 2*config.max_sys_trace + 2:]} #create the entry dictionary with the current and target segments, where the target segment has only the config.ny columns
        else:
            # generate random entries
            entry = self.entries[idx % len(self.entries)].copy()

            obs = entry.pop("obs")
            L = obs.shape[-2]
            if config.dataset_typ in ["unifA", "noniid", "upperTriA", "upperTriA_gauss", "rotDiagA", "rotDiagA_unif", "rotDiagA_gauss", "gaussA", "gaussA_noscale", "config.single_system", "cond_num", "ident", "ortho"]:
                entry["current"] = np.take(obs, np.arange(L - 1), axis=-2) #current observation
                entry["target"] = np.take(obs, np.arange(1, L), axis=-2) #true value of target observation at the next instance
            else:
                raise NotImplementedError(f"{config.dataset_typ} is not implemented")

        torch_entry = dict([
            (k, (torch.from_numpy(a) if isinstance(a, np.ndarray) else a).to(torch.float32))
            for k, a in entry.items()])
        return torch_entry
