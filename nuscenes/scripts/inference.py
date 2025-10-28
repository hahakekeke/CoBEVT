import os
import glob
import time
import torch
import pytorch_lightning as pl
import wandb
from pathlib import Path
import hydra

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone

# =========================
# 자동 최신 체크포인트 찾기
# =========================
def get_latest_ckpt(outputs_dir="/content/CoBEVT/outputs"):
    ckpt_list = glob.glob(os.path.join(outputs_dir, "*/checkpoints/model.ckpt"))
    if not ckpt_list:
        raise FileNotFoundError(f"No checkpoint found in {outputs_dir}")
    ckpt_list.sort(key=os.path.getmtime)
    latest_ckpt = ckpt_list[-1]
    print("Using checkpoint:", latest_ckpt)
    return latest_ckpt

# =========================
# Lightning Callback 정의
# =========================
class InferenceTimeCallback(pl.Callback):
    def __init__(self, data_module, ckpt_path=None):
        super().__init__()
        self.data_module = data_module
        self.ckpt_path = ckpt_path

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        dataloader = self.data_module.val_dataloader()  # test_dataloader 대신 val_dataloader 사용
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
        import wandb
        wandb.log({"avg_inference_time_sec": avg_time, "epoch": trainer.current_epoch})
        pl_module.train()


# =========================
# 추론 실행 함수
# =========================
def run_inference(cfg, ckpt_path):
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # 모델/데이터 준비
    model_module, data_module, viz_fn = setup_experiment(cfg)
    model_module.backbone = load_backbone(ckpt_path)
    model_module.eval()
    model_module.cuda()

    # W&B 초기화
    wandb.init(project=cfg.experiment.project, name="inference_run", config=cfg)

    # Callback으로 추론시간 측정
    test_loader = data_module.test_dataloader()
    callbacks = [InferenceTimeCallback(test_loader)]

    # Trainer로 epoch 반복만 수행 (fit 대신 limit_train_batches=0)
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=False,  # 이미 W&B init 했으므로
        callbacks=callbacks,
        limit_train_batches=0  # 학습 skip
    )
    trainer.fit(model_module, datamodule=data_module)

    wandb.finish()

# =========================
# Hydra config 실행
# =========================
CONFIG_PATH = '/content/CoBEVT/nuscenes/config'
CONFIG_NAME = 'config.yaml'

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    ckpt_path = get_latest_ckpt()
    run_inference(cfg, ckpt_path)

if __name__ == "__main__":
    main()
