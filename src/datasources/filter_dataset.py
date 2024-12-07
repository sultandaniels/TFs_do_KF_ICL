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
