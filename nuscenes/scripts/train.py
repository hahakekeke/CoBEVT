'''

from pathlib import Path
import logging
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback


log = logging.getLogger(__name__)

CONFIG_PATH = '/content/CoBEVT/nuscenes/config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()


'''


from pathlib import Path
import logging
import time
import torch
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback
import wandb

log = logging.getLogger(__name__)

CONFIG_PATH = '/content/CoBEVT/nuscenes/config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))
    log.info(f'Searching {save_dir}.')
    if not checkpoints:
        return None
    log.info(f'Found {checkpoints[-1]}.')
    return checkpoints[-1]


# =========================
# Callback: 매 epoch마다 추론시간 측정
# =========================
class InferenceTimeCallback(Callback):
    def __init__(self, data_module, ckpt_path=None):
        super().__init__()
        self.data_module = data_module
        self.ckpt_path = ckpt_path

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        dataloader = self.data_module.test_dataloader()
        total_time = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                start = time.time()
                _ = pl_module(batch)
                end = time.time()
                total_time += (end - start)
                total_batches += 1
        avg_time = total_time / total_batches if total_batches > 0 else 0.0
        print(f"[Epoch {trainer.current_epoch}] Average inference time per batch: {avg_time:.4f} sec")
        wandb.log({"avg_inference_time_sec": avg_time, "epoch": trainer.current_epoch})
        pl_module.train()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),
        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg),
        InferenceTimeCallback(data_module, ckpt_path)  # ✅ 추가
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
