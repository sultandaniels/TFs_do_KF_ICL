from torch.utils.data import Dataset
from dyn_models.filtering_lti import *
from core import Config
import torch
import pickle


config = Config()


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
        if config.multi_sys_trace:
            context_len = config.n_positions
            segments = np.zeros((context_len, config.ny)) #initialize the segments array
            print('segments.shape', segments.shape)
            possible_space = context_len - 2 #the possible space for the system traces
            while possible_space > 0:
                print('\npossible_space', possible_space)
                segment_len = np.random.randint(1, possible_space + 1) #randomly sample a segment length between 1 and the possible space
                print('\nsegment_len', segment_len)

                # select a random integer between 0 and len(self.entries)
                sys_trace_ind = np.random.choice(config.num_tasks) #randomly sample a number between 0 and num_tasks (random system index)
                print('\nsys_trace_ind', sys_trace_ind)
                #get obs from the system trace corresponding to sys_trace_ind
                sys_trace_obs = self.entries[sys_trace_ind]["obs"]
                print('sys_trace_obs.shape', sys_trace_obs.shape)
                random_start = np.random.randint(0, sys_trace_obs.shape[-2] - segment_len) #randomly sample a starting index for each segment
                print('random_start', random_start)
                segment = sys_trace_obs[..., random_start:random_start + segment_len, :]
                print('segment.shape orig:', segment.shape)
                print_matrix(segment, 'segment orig')

                # Create the special tokens
                start_token = (100 * (sys_trace_ind + 1)) * np.ones((1, segment.shape[1]))
                end_token = (100 * (sys_trace_ind + 1) + 1) * np.ones((1, segment.shape[1]))

                tok_seg_len = segment_len + 2
                segment = np.concatenate([start_token, segment, end_token], axis=0)
                print('segment.shape post special tokens', segment.shape)
                print_matrix(segment, 'segment post special tokens')

                segments[context_len - possible_space:context_len - possible_space + tok_seg_len, :] = segment
                entry = {"current": segments[:-1, :], "target": segments[1:, :]}
                print('entry["current"].shape', entry["target"].shape)


                possible_space -= tok_seg_len #update the possible space for the next iteration


            ###################################################################################
            # while segment_len >= 1:
            #     sys_trace_num = np.random.choice(config.num_tasks) + 1 #randomly sample a number between 1 and num_tasks (random number of system traces)
            #     print('\nsys_trace_num', sys_trace_num)

            #     eff_context_len = context_len - (2 * sys_trace_num) #effective context length after accounting for the special tokens
            #     print('\n eff_context_len', eff_context_len)

            #     segment_len = eff_context_len // sys_trace_num #length of each segment (floor division)
            #     print('segment_len', segment_len)


            # # get sys_trace_num random numbers between 0 and len(self.entries) inclusive and name them sys_inds
            # sys_inds = np.random.choice(len(self.entries), sys_trace_num, replace=False)
            # print('sys_inds', sys_inds)

            # # get the sys_trace_entries corresponding to the sys_inds
            # sys_trace_entries = [self.entries[i] for i in sys_inds]

            # # for each sys_trace_entry, get their observations
            # sys_trace_obs = [entry["obs"] for entry in sys_trace_entries]

            # sys_count = 1
            # segments = np.zeros((context_len, sys_trace_obs[0].shape[-1])) #initialize the segments array
            # for obs in np.random.permutation(sys_trace_obs): #randomly interleave the system traces
            #     print('obs.shape', obs.shape)
            #     random_start = np.random.randint(0, obs.shape[-2] - segment_len) #randomly sample a starting index for each segment
            #     print('random_start', random_start)
            #     segment = obs[..., random_start:random_start + segment_len, :]
            #     print('segment.shape orig:', segment.shape)
            #     print_matrix(segment, 'segment orig')


            #     # Create the special tokens
            #     start_token = (100 * sys_count) * np.ones((1, segment.shape[1]))
            #     end_token = (100 * sys_count + 1) * np.ones((1, segment.shape[1]))

            #     # Concatenate the special tokens to the beginning and end of the segment
            #     segment = np.concatenate([start_token, segment, end_token], axis=0)
            #     print('segment.shape post special tokens', segment.shape)
            #     print_matrix(segment, 'segment post special tokens')

            #     tok_seg_len = segment_len + 2
            #     segments[(sys_count - 1) * tok_seg_len:sys_count * tok_seg_len, :] = segment
            #     sys_count += 1

            # print('segments.shape', segments.shape)
            # print_matrix(segments, "segments")

            # entry = {"current": segments[:-1, :], "target": segments[1:, :]}
            # print('entry["current"].shape', entry["target"].shape)

            
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
        
        print("torch_entry", torch_entry)
        raise NotImplementedError("stop here")
        return torch_entry
