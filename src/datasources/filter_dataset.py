from torch.utils.data import Dataset
from dyn_models.filtering_lti import *
from core import Config
import torch
import pickle
import pdb

config = Config()


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
            sys_trace_num = config.multi_sys_trace_num

            eff_context_len = context_len - (2 * sys_trace_num) #effective context length after accounting for the special tokens
            print('eff_context_len', eff_context_len)

            segment_len = eff_context_len // sys_trace_num #length of each segment (floor division)
            print('segment_len', segment_len)

            # get sys_trace_num random entries from self.entries
            sys_trace_entries = [self.entries[i] for i in np.random.choice(len(self.entries), sys_trace_num, replace=False)] #randomly sample sys_trace_num entries from the dataset

            # for each sys_trace_entry, get their observations
            sys_trace_obs = [entry["obs"] for entry in sys_trace_entries]

            sys_count = 1
            segments = np.zeros((context_len, sys_trace_obs[0].shape[-1])) #initialize the segments array
            for obs in sys_trace_obs:
                print('obs.shape', obs.shape)
                random_start = np.random.randint(0, obs.shape[-2] - segment_len) #randomly sample a starting index for each segment
                print('random_start', random_start)
                segment = obs[..., random_start:random_start + segment_len, :]
                print('segment.shape', segment.shape)
                print('segment', segment)
                #append a special token to the beginning and end of each segment
                segment = np.concatenate([(100*sys_count)*np.ones((segment.shape[0], 1, segment.shape[2])), segment, (100*sys_count + 1)*np.ones((segment.shape[0], 1, segment.shape[2]))], axis=1)
                print('segment.shape', segment.shape)
                print('segment', segment)
                segments[(sys_count - 1) * segment_len:sys_count * segment_len, :] = segment
                sys_count += 1

            print('segments.shape', segments.shape)
            print('segments', segments)

            entry = {"current": segments[:-1, :], "target": segments[1:, :]}
            print('entry["current"].shape', entry["target"].shape)

            
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
