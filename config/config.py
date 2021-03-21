import os
import argparse
import torch

HOME_DIR = os.path.expanduser("~")

class Config:
    # id
    # data
    default_config = dict(
        exp_ver_name='v0',
        randseed=42,
        dry_run=False,
        does_ray_tuning=False,
        root_dir=os.getcwd(),
        data_dir=os.path.join(HOME_DIR, "data", "ocean", "ZZ"),
        data_name="Chl.h5",
        small_data_name="Chl_small.h5",
        log_dir="logs",
        checkpoint_dir="checkpoints",
        train_val_split_ratio=0.8,
        num_workers_train_loader=6,
        num_workers_val_loader=6,

        # meta settings 
        batch_size=16,
        shuffle=True,

        # optimisation
        max_epoches=10,
        learning_rate=1e-4,
        weight_decay=0,
        optimiser="Adam",

        # model settings
        latent_dim=16)
    
    def __init__(self, args):
        self._yaml_config = self._from_yaml(args.base_config_file)

        self._set_values(args) 
        self._setup_dirs()
        self._derive_settings()
        self._check_parameters()

    @staticmethod    
    def add_argparse_args(parser):
        # - identify the experiment for logging and checkpoint saving
        parser.add_argument('exp_model_name', type=str,
            metavar="EXPERIMENT_MODEL", help="experiment name")
        parser.add_argument('--exp-ver-name', metavar="VER", 
            type=str,
            help="experiment version, default v0")
        # program hyper-params
        parser.add_argument('-n', '--dry-run', action='store_true')
        parser.add_argument('-y', '--base-config-file', type=str, 
            help="YAML exper-specs, can be used as base settings "\
                "with modifications specified here.")
        parser.add_argument('--does-ray-tuning', default=None, 
            action='store_true',
            help="automatic run by ray[tune] to determine h-params")
        parser.add_argument('-g', '--gpus', type=int,
            action='extend', nargs="+",
            help="GPUs to use, e.g. -g 0 1 2")
        parser.add_argument('--root-dir', type=str,
            help="dir of logs and checkpoints default: PWD")
        parser.add_argument('--data-dir', type=str, 
            help="dir contains the dataset --data-name")
        parser.add_argument('--log-dir', type=str, 
            help="root-log dir, loggers to manage sub-dirs of experiments "
                "default $root_dir/logs")
        parser.add_argument('--checkpoint-dir', type=str, 
            help="root-checkpoint dir, "
            "logs will be saved to .../full-exp-name/ "
            "default: $root_dir/checkpoints")
        parser.add_argument('-c', '--resume-from-checkpoint', type=str, 
            metavar='WARM-START-CHECKPOINT',
            help='FULL PATH to checkpoint file, will supersede all other settings')
        parser.add_argument('--data-name', type=str, 
            help="file name, e.g. Chl.h5")
        parser.add_argument('--small_data-name', type=str, 
            help="file name, e.g. Chl_small.h5")
        parser.add_argument('--does-use-small-data', action="store_true",
            default=None, help="fast prototyping using small dataset")

        # high-level training process hyper-params
        parser.add_argument('--randseed', type=int,
            help="random seed, default 42")
        parser.add_argument('--train-val-split-ratio', type=float, 
            help="train in (train + val) split")
        parser.add_argument('--num-workers-train-loader', type=int,
            help="#. train loader workers")
        parser.add_argument('--num-workers-val-loader', type=int,
            help="#. train loader workers")
        return parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        pp = parent_parser
        pp.add_argument('--batch-size', type=int, help="samples in a batch")
        pp.add_argument('--max-epoches', type=int, help="training epoches")
        pp.add_argument('--learning-rate', type=float, help="lr")
        pp.add_argument('--weight-decay', type=float, help="weight decay")
        pp.add_argument('-o', '--optimiser', type=str, help="optimiser")
        pp.add_argument('-d', '--latent-dim', type=int, help="internal dim")
        return pp

    @staticmethod
    def add_tuning_args(parent_parser):
        parent_parser.add_argument('--max-tune-epoches', type=int,
            default=99, help="epoches in tuning rounds")

        return parent_parser

    def _set_values(self, args):
        print("== Set configurable hyper-parameters ==")
        # first pass cli
        for a, v in vars(args).items():
            fv = self._yaml_config.get(a)
            if v is not None:
                print(f"Setting {a} to cli-assigned value {v}")
                self.__setattr__(a, v)
                if fv:
                    print(f"\tOverwriting config-file value {fv}")
                
                if a == 'structure_index':
                    # todo: try different structures
                    _ = """
                    for _k in ["patch_rows", "patch_cols", \
                        "patch_rows_2", "patch_cols_2"]:
                        try:
                            del Config.default_config[_k]
                        except:
                            pass
                        try:
                            del self._yaml_config[_k]
                        except:
                            pass
                    """

        # second pass config file
        for a, v in vars(args).items():
            fv = self._yaml_config.get(a)
            if v is None and fv is not None:
                print(f"Setting {a} to config-file value {fv}")
                self.__setattr__(a, fv)

                if a == 'structure_index':
                    _ = """
                    # delete default values for patch_rows/cols/_2
                    for _k in ["patch_rows", "patch_cols", \
                        "patch_rows_2", "patch_cols_2"]:
                        try:
                            del Config.default_config[_k]
                        except:
                            pass"""

        # 3rd pass, default
        for a, v in vars(args).items():
            fv = self._yaml_config.get(a)
            if v is not None or fv is not None:
                continue
            if a in Config.default_config.keys():
                dv = Config.default_config[a]
                print(f"Setting {a} to default value {dv}")
                self.__setattr__(a, dv)
            else:
                print(f"None valid value found for {a}")
                self.__setattr__(a, None)

    def _setup_dirs(self):
        osp = os.path
        full_pth = lambda s: osp.abspath(osp.expanduser(s))
        if not osp.isabs(self.root_dir):
            self.root_dir = full_pth(self.root_dir)

        if not osp.isabs(self.data_dir):
            self.data_dir = osp.join(self.root_dir, self.data_dir)
        assert os.path.exists(os.path.join(self.data_dir, self.data_name)), \
            f"Cannot find data file {self.data_name} in {self.data_dir}"
        assert os.path.exists(
                os.path.join(self.data_dir, self.small_data_name)),\
            f"Cannot find data file {self.data_name} in {self.data_dir}"

        if not osp.isabs(self.log_dir):
            self.log_dir = osp.join(self.root_dir, self.log_dir)

        if not osp.isabs(self.checkpoint_dir):
            self.checkpoint_dir = osp.join(self.root_dir, self.checkpoint_dir)

        print("== Directories ==")
        print(f"         data:{self.data_dir}")
        print(f"          log:{self.log_dir}")
        print(f"  checkpoints:{self.checkpoint_dir}") 

    def _derive_settings(self):
        self.exp_full_name = '_'.join([
            self.exp_model_name, 
            self.data_name, 
            self.exp_ver_name])
        self.checkpoint_dir = os.path.join(
            self.checkpoint_dir, self.exp_full_name)
        # todo derive VAE settings from a few parameters

    def _check_parameters(self):
        # sanity check goes here
        pass

    def _from_yaml(self, yaml_fname:str) -> dict:
        if yaml_fname:
            if os.path.isabs(yaml_fname):
                self.base_config_file = yaml_fname
            else:
                self.base_config_file = os.path.abspath(
                    os.path.expanduser(yaml_fname))

            print(f"Loading base settings from {self.base_config_file}")
            d = load_yaml(yaml_fname)
        else:
            d = {}
        return d