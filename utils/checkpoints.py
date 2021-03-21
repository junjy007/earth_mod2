import os
import pytorch_lightning as pl

# Checkpoint every n steps
# https://github.com/PyTorchLightning/pytorch-lightning/issues/2534
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, *args):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}" + "_{epoch:}_{global_step:}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def get_checkpoints(
    experiment_name: str = "MyExperiment",
    checkpoint_dir: str = "pl_checkpoints",
    every_n_steps: int = 20000,
    monitor: str = 'val_loss',
    mode: str = 'min'):
    checkpoint_callback1 = pl.callbacks.ModelCheckpoint(
        verbose=True,
        monitor=monitor,
        dirpath=checkpoint_dir,
        filename=f'{experiment_name}'+'-{epoch:02d}-{val_loss:.2f}',
        save_last=True,
        save_top_k=3,
        mode=mode
    )
    checkpoint_callback2 = CheckpointEveryNSteps(
        every_n_steps, prefix=experiment_name+'_ns')
    return [checkpoint_callback1, checkpoint_callback2]