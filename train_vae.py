from utils.common import *
from config.config import Config
from utils.experiment import CholDataModule, LitVAE
from pytorch_lightning import Trainer, seed_everything
from utils.checkpoints import get_checkpoints
from argparse import ArgumentParser
import os
from config.config import Config
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from copy import deepcopy

def main_tune(base_args):
    # ray.init(log_to_driver=False)
    tune_config = {
        "learning_rate": tune.loguniform(5e-6, 1e-3),
        "weight_decay": tune.choice([0.0, 1e-3, 1e-2, 0.1]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "latent_dim": tune.choice([2, 3, 8, 16, 32, 128, 256, 512])
    }

    scheduler = ASHAScheduler(
        max_t=base_args.max_tune_epoches,
        grace_period=3,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            "learning_rate", "weight_decay", 
            "batch_size", "latent_dim"],
        metric_columns=["val_lossR", "loss", 
            "Reconstruction_Loss", "training_iteration"])
    

    analysis = tune.run(
        tune.with_parameters(tune_train, base_arg=base_args),
        resources_per_trial={
            "cpu": 12,
            "gpu": 1.0,
        },
        metric="val_lossR",
        mode="min",
        config=tune_config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_vae_chol"
    )

    print("Best hyperparameters found were: ", analysis.best_config)


def tune_train(tune_config:dict, base_arg):
    """
    Wrapper for ray.tune to adjust necessary params
    """
    # note tune config
    arg = deepcopy(base_arg)
    arg.learning_rate = tune_config['learning_rate']
    arg.weight_decay = tune_config['weight_decay']
    arg.batch_size = tune_config['batch_size']
    arg.latent_dim = tune_config['latent_dim']
    arg.does_ray_tuning = True

    cfg = Config(arg)

    main(cfg, callbacks=[
        TuneReportCallback(
            metrics=["val_lossR",], 
            on="validation_end")
        ]
    )    

def main(cfg, **kwargs):
    # book-keeping
    # todo: just specify checkpoints, not if ray is running
    if kwargs.get('callbacks'):
        callbacks = kwargs['callbacks']
    else:
        callbacks = []

    if cfg.does_ray_tuning:
        trainer_args = {
            'default_root_dir': cfg.root_dir,
            'max_epochs': cfg.max_epoches,
            'deterministic': True,
            'callbacks': callbacks,
            'gpus': cfg.gpus
        }
    else:
        callbacks += get_checkpoints(
            experiment_name=cfg.exp_full_name,
            checkpoint_dir=os.path.join("pl_checkpoints", cfg.exp_full_name),
            monitor='val_lossR', mode='min')

        logger = pl.loggers.TensorBoardLogger(
            "logs", name=cfg.exp_full_name)

        trainer_args = {
            'default_root_dir': cfg.root_dir,
            'max_epochs': cfg.max_epoches,
            'deterministic': True,
            'logger': logger,
            'callbacks': callbacks,
            'gpus': cfg.gpus
        }

    seed_everything(cfg.randseed)
    chol_data = CholDataModule(cfg)
    exper = LitVAE(cfg=cfg)
    # if warm start, 
    # trainer_args['resume_from_checkpoint'] = "pl_checkpoints/last.ckpt"
    trainer = Trainer(**trainer_args)
    trainer.fit(exper, chol_data)

if __name__ == '__main__':
    parser = ArgumentParser("Chol Data Analyser")
    parser = Config.add_argparse_args(parser)
    parser = Config.add_model_specific_args(parser)
    parser = Config.add_tuning_args(parser)
    args = parser.parse_args()

    # args = parser.parse_args()
    if args.does_ray_tuning:
        main_tune(args)
    else:
        cfg = Config(args) 
        main(cfg)
