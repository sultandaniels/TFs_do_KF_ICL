from torch.utils.data import Dataset
from dyn_models.filtering_lti import *
from core import Config
import torch
import pickle
from linalg_helpers import print_matrix


config = Config()

def generate_zipf_integer(n, a):
    """
    Generate integer number between 1 and n (inclusive) from a Zipf's power law distribution.

    Parameters:
    n (int): The upper limit (inclusive) for the range of integers.
    a (float): The parameter of the Zipf distribution (a > 1).

    Returns:
    np.ndarray: An array of integers between 0 and n.
    """
    # Generate samples from a Zipf distribution
    sample = np.random.zipf(a, 1)

    # Clip the samples to the desired range (1 to n)
    sample = np.clip(sample, 0, n)

    return sample[0]



def print_matrix(matrix, name):
    """
    Print a matrix in a readable format.
    
    Parameters:
    matrix (np.ndarray): The matrix to print.
    name (str): The name of the matrix.
    """
    print(f"Matrix {name}:")
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            print(f"{matrix[i, j]:>10.4f}", end=" ")
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
        # create an array of zeros with the same number of columns as the segment, but with a 1 in the column corresponding to the 2*system name
        start_token = np.zeros((1, segment.shape[1]))
        start_token[0, 2*sys_name] = 1
        end_token = np.zeros((1, segment.shape[1]))
        end_token[0, 2*sys_name + 1] = 1
    else:
        raise ValueError(f"Special token style {style} has not been implemented.")
    
    return start_token, end_token

def populate_traces(n_positions, ny, num_tasks, entries, max_sys_trace, test=False):
    sys_choices = [] #list that will hold the order of the system choices for the trace
    seg_starts = []


    sys_names = np.arange(max_sys_trace) #system names
    #randomly shuffle the system names
    np.random.shuffle(sys_names)
    
    sys_in_trace = generate_zipf_integer(max_sys_trace,2) #number of systems to include in the context

    #uniformly at random select sys_in_traces numbers between 0 and num_tasks without replacement for the system indices
    sys_inds = np.random.randint(0, num_tasks, sys_in_trace).tolist()

    #create a tuple that matches the system names to the system indices
    sys_dict = {}
    for i in range(len(sys_inds)):
        sys_dict[sys_inds[i]] = sys_names[i]
        

    seg_lens = [] #initialize the list of segment lengths
    while sum(seg_lens) < n_positions:
        seg_lens = np.random.binomial(n_positions, 1/sys_in_trace, size=10*sys_in_trace) #randomly sample segment lengths for the trace segments (p = 1/sys_in_trace, so that when sys_in_trace = 1, there will only be one segment)

    context_len = n_positions + 1
    segments = np.zeros((context_len, ny + 2*max_sys_trace + 1)) #initialize the segments array
    segments[0, 2*max_sys_trace] = 1 #set the start token for the first segment

    #initialize a dictionary to hold the next starting index for each system trace
    next_start = {sys_ind: 0 for sys_ind in sys_inds} 

    seg_start = 1 #initialize the starting index for the segment at 1 to account for the start token
    for seg_len in seg_lens:
        seg_starts.append(seg_start)

        if seg_start > 1:
            old_sys_ind = sys_ind

        #pick a random system index
        sys_ind = np.random.choice(sys_inds)
        sys_choices.append(sys_ind) #add the system index to the list of system choices

        sys_inds.remove(sys_ind)

        if seg_start > 1:
            sys_inds.append(old_sys_ind) #replace the old sys_ind in the list (this ensures the same system isn't picked twice in a row)

        #get obs from the system trace corresponding to sys_trace_ind
        if test:
            sys_trace_obs = entries[sys_ind]
        else:
            sys_trace_obs = entries[sys_ind]["obs"]

        if next_start[sys_ind] + seg_len > sys_trace_obs.shape[0]: #if the next starting index plus the segment length is greater than the length of the trace
            if next_start[sys_ind] >= sys_trace_obs.shape[0]: #if the next starting index is greater than the length of the trace, skip to the next trace
                continue
            else:
                segment = sys_trace_obs[next_start[sys_ind]:, :] #get the segment from the next starting index to the end of the trace
                seg_len = segment.shape[0] #update the segment length to the length of the segment
        else:
            segment = sys_trace_obs[next_start[sys_ind]:next_start[sys_ind] + seg_len, :] #get the segment from the next starting index to the next starting index plus the segment length
        
        next_start[sys_ind] += seg_len #update the next starting index for the trace from this system index 

        #concatenate 2*max_sys_trace + 1 columns of zeros to the segment
        zeros = np.zeros((segment.shape[0], 2*max_sys_trace + 1))
        segment = np.concatenate((zeros, segment), axis=1)

        start_paren, end_paren = special_tokens(segment, sys_dict[sys_ind], style="zeros") #get the special tokens for the segment

        segment = np.concatenate([start_paren, segment, end_paren], axis=0) #concatenate the special tokens to the segment

        if seg_start + seg_len + 2 > context_len:
            #truncate the segment if it is too long so that it fits in the context
            segment = segment[:context_len - seg_start, :]
            break

        tok_seg_len = segment.shape[0]

        segments[seg_start:seg_start + tok_seg_len, :] = segment #add the segment to the segments array

        if seg_start + tok_seg_len == context_len:
            break

        seg_start += tok_seg_len #update the starting index for the next segment

    return segments, sys_choices, sys_dict, seg_lens, seg_starts


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
            segments, sys_choices, sys_dict, seg_lens, seg_starts = populate_traces(config.n_positions, config.ny, config.num_tasks, self.entries, config.max_sys_trace)
            entry = {"current": segments[:-1, :], "target": segments[1:, 2*config.max_sys_trace + 1:]} #create the entry dictionary with the current and target segments, where the target segment has only the ny columns
        else:
            # generate random entries
            entry = self.entries[idx % len(self.entries)].copy()

            obs = entry.pop("obs")
            L = obs.shape[-2]
            if config.dataset_typ in ["unifA", "noniid", "upperTriA", "upperTriA_gauss", "rotDiagA", "rotDiagA_unif", "rotDiagA_gauss", "gaussA", "gaussA_noscale", "single_system", "cond_num"]:
                entry["current"] = np.take(obs, np.arange(L - 1), axis=-2) #current observation
                entry["target"] = np.take(obs, np.arange(1, L), axis=-2) #true value of target observation at the next instance
            else:
                raise NotImplementedError(f"{config.dataset_typ} is not implemented")

        torch_entry = dict([
            (k, (torch.from_numpy(a) if isinstance(a, np.ndarray) else a).to(torch.float32))
            for k, a in entry.items()])
        return torch_entry
