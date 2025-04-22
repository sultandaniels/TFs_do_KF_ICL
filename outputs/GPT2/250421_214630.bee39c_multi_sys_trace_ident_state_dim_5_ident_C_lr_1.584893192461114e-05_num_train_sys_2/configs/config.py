import yaml
import os
import zipfile
import argparse

import coloredlogs
from utils import log_info
from utils import Singleton
from utils import set_seed

# /checkpoints/step=10000.ckpt

class Config(object, metaclass=Singleton):
    ckpt_path = ""
    seed = 0
    fully_reproducible = False

    # Dataset settings
    num_tasks = 2#40000 #number of training systems
    num_val_tasks = 1#100 #number of test systems
    dataset_typ = "ident" #"unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"upperTriA_gauss" #"ident" #"ortho" #"ortho_haar"
    max_cond_num = 100
    distinct_cond_nums = 10
    val_dataset_typ = "ident" #"unifA" #"gaussA" #"gaussA_noscale" #"rotDiagA" #"rotDiagA_unif" #"rotDiagA_gauss" #"upperTriA" #"single_system" #"cond_num" #"ident" #"ortho" #"ortho_haar"
    C_dist = "_ident_C" #"_unif_C" #"_gauss_C" #"_gauss_C_large_var" #"_single_system" #"upperTriA_gauss" #"_ident_C"
    nx = 5
    ny = 5
    n_noise = 1
    num_traces = {"train": 1, "val": 1}
    changing = False #used only for plotting

    #experiment settings
    multi_sys_trace = True #have multiple systems in a single trace
    max_sys_trace = min(25, num_tasks) #maximum number of systems in a trace
    single_system = False #only use a single system in the test trace
    zero_cut = False #no cuts in the trace interleaving
    needle_in_haystack = False #run needle in haystack tests
    needle_final_seg_extended = False #extend the final segment of the needle in haystack test
    datasource="val" #"val" #"train" #"train_systems" #which dataset to use for the needle in haystack tests
    num_sys_haystack = 19 #1 #2 #3 #4 #9 #14 #19 #number of systems in the haystack
    len_seg_haystack = 10 #123 #82 #61 #48 #23 #15 #10 #length of a haystack segment
    num_haystack_examples = 200 #number of haystack examples to generate
    num_test_traces_configs = num_sys_haystack if needle_in_haystack and (not needle_final_seg_extended) else (1 if needle_in_haystack and needle_final_seg_extended else (num_val_tasks if zero_cut else 1)) #number of test traces configurations to generate

    # Training settings
    devices=[0] #which GPU
    train_steps = 3#1008000 #number of training steps (27000x3 = 81000 effective single GPU iterations)      (num_tasks*num_traces[train])/batch_size
    num_epochs = 1 #1000 #minimum number of epochs to train for
    train_int = 1 #number of steps between logging (train interval)
    use_true_len = False #Flag for a dataset length to be num_tasks
    batch_size = 2 #usually 512 (~35GB) tune this to fit into GPU memory
    acc_grad_batch = 1 #number of batches to accumulate gradients over
    train_data_workers = 128 #set to 1 (check if it changes the speed of the training process)
    test_batch_size = 256
    test_data_workers = 1 #keep at 1

    # Model settings
    model_type = "GPT2" #"GPT2" #"transfoXL" #"olmo"
    use_pos_emb = True #use positional embeddings
    n_positions = 250 #500 for extended OLS #250 #context length
    n_embd = 128
    n_layer = 12
    n_head = 8
    n_dims_in = int(ny + (2*max_sys_trace) + 2) if multi_sys_trace else ny #input dimension is the observation dimension #input dimension is the observation dimension + special token parentheses + special start token + payload identifier
    n_dims_out = 5  #(IMPORTANT TO KEEP THIS AT 5 FOR NOW) TODO: this used to be 10 but needs to be fixed to match lin_sys.yaml


    #transfoXL specific
    d_model = 512
    # d_inner = 2048
    # cutoffs = [20000, 40000, 200000]
    # div_val = 4
    # mem_len = 1600
    # same_length = True
    # clamp_len = 1000

    # Optimizer parameters
    learning_rate = 1.584893192461114e-05 #1.9054607179632464e-05
    weight_decay = 1e-2

    # Gradient Clipping
    gradient_clip_algorithm = 'norm'  # 'norm' or 'value'
    gradient_clip_val = 1.0

    def __new__(cls):
        __instance = super().__new__(cls)
        cls.__filecontents = cls.__get_config_file_contents()
        cls.__immutable = True
        return __instance

    def import_yaml(self, yaml_path, strict=True):
        """Import YAML config to over-write existing config entries."""
        assert os.path.isfile(yaml_path)
        assert not hasattr(self.__class__, '__imported_yaml_path')
        log_info('Loading ' + yaml_path)
        with open(yaml_path, 'r') as f:
            yaml_string = f.read()
        self.import_dict(
            yaml.load(yaml_string, Loader=yaml.FullLoader), strict=strict)
        self.__class__.__imported_yaml_path = yaml_path
        self.__class__.__filecontents[os.path.basename(
            yaml_path)] = yaml_string

    def override(self, key, value):
        self.__class__.__immutable = False
        if not hasattr(self, key):
            # raise Exception("Tried to override non-existing key: " + key)
            print("Creating new key: " + key)
        setattr(self, key, value)
        self.__class__.__immutable = True

    def import_dict(self, dictionary, strict=True):
        """Import a set of key-value pairs from a dict to over-write existing config entries."""
        self.__class__.__immutable = False
        for key, value in dictionary.items():
            if strict is True:
                if not hasattr(self, key):
                    raise ValueError('Unknown configuration key: ' + key)
                if type(getattr(self, key)) is float and type(value) is int:
                    value = float(value)
                else:
                    assert type(getattr(self, key)) is type(
                        value), f"{key}, {type(getattr(self, key))}, {type(value)}"
                if not isinstance(getattr(self, key), property):
                    setattr(self, key, value)
            else:
                if hasattr(Config, key):
                    if not isinstance(getattr(self, key), property):
                        setattr(self, key, value)
                else:
                    setattr(self, key, value)
        self.__class__.__immutable = True

    @classmethod
    def __get_config_file_contents(cls):
        """Retrieve and cache default and user config file contents."""
        out = {}
        for relpath in ['config.py']:
            path = os.path.relpath(os.path.dirname(__file__) + '/' + relpath)
            assert os.path.isfile(path)
            with open(path, 'r') as f:
                out[os.path.basename(path)] = f.read()
        return out

    def get_all_key_values(self):
        return dict([
            (key, getattr(self, key))
            for key in dir(self)
            if not key.startswith('_Config')
               and not key.startswith('__')
               and not callable(getattr(self, key))
        ])

    def get_full_yaml(self):
        return yaml.dump(self.get_all_key_values())

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert os.path.isdir(target_base_dir)

        # Write config file contents
        target_dir = target_base_dir + '/configs'
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        outputs = {  # Also output flattened config
            'combined.yaml': self.get_full_yaml(),
        }
        outputs.update(self.__class__.__filecontents)
        for fname, content in outputs.items():
            fpath = os.path.relpath(target_dir + '/' + fname)
            with open(fpath, 'w') as f:
                f.write(content)
                log_info('Written %s' % fpath)

        # Copy source folder contents over
        target_path = os.path.relpath(target_base_dir + '/src.zip')
        source_path = os.path.relpath(os.path.dirname(__file__) + '/../')

        def filter_(x):
            return x.endswith('.py') or x.endswith('.yaml')  # noqa

        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = os.path.join(root, file_or_dir)
                    if os.path.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            os.path.join(root, file_or_dir),
                            os.path.relpath(os.path.join(root, file_or_dir),
                                            os.path.join(source_path, os.path.pardir)))
        log_info('Written source folder to %s' % os.path.relpath(target_path))

    def __setattr__(self, name, value):
        """Initial configs should not be overwritten!"""
        if self.__class__.__immutable:
            raise AttributeError('Config instance attributes are immutable.')
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        """Initial configs should not be removed!"""
        if self.__class__.__immutable:
            raise AttributeError('Config instance attributes are immutable.')
        else:
            super().__delattr__(name)

    def __eq__(self, other):
        if isinstance(other, Config):
            other = other.get_all_key_values()
        elif type(other) != dict:
            raise Exception('Config can only be compared to Config or dict')
        return self.get_all_key_values() == other

    def __convert_cli_arg_type(self, key, value):
        config_type = type(getattr(self, key))
        if config_type == bool:
            if value.lower() in ('true', 'yes', 'y') or value == '1':
                return True
            elif value.lower() in ('false', 'no', 'n') or value == '0':
                return False
            else:
                raise ValueError(
                    'Invalid input for bool config "%s": %s' % (key, value))
        else:
            return config_type(value)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-v', type=str, help='Desired logging level.', default='info',
                            choices=['debug', 'info', 'warning', 'error', 'critical'])
        parser.add_argument('config_yaml', type=str, nargs='*',
                            help=('Path to config in YAML format. '
                                  'Multiple configs will be parsed in the specified order.'))
        for key in dir(self):
            if key.startswith('_DefaultConfig') or key.startswith('__'):
                continue
            if key in vars(type(self)) and isinstance(vars(type(self))[key], property):
                continue
            value = getattr(self, key)
            value_type = type(value)
            arg_type = value_type
            if value_type == bool:
                # Handle booleans separately, otherwise arbitrary values become `True`
                arg_type = str
            if callable(value):
                continue
            parser.add_argument('--' + key.replace('_', '-'), type=arg_type, metavar=value,
                                help='Expected type is `%s`.' % value_type.__name__)
        args = parser.parse_args()

        # Set logger format and verbosity level
        coloredlogs.install(
            datefmt='%d/%m %H:%M:%S',
            fmt='%(asctime)s %(levelname)s %(message)s',
            level=args.v.upper(),
        )

        # Parse configs in order specified by user
        for yaml_path in args.config_yaml:
            self.import_yaml(yaml_path)

        # Apply configs passed through command line
        self.import_dict({
            key.replace('-', '_'): self.__convert_cli_arg_type(key, value)
            for key, value in vars(args).items()
            if value is not None and hasattr(self, key)
        })
        set_seed(self.seed, self.fully_reproducible)
