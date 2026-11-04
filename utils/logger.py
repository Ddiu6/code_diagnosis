"""
Simple logging utilities for training.
"""

import os
import time


class Logger:
    """Simple dual-output logger (console + file)."""

    def __init__(self, log_dir: str, filename: str = "train.log"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self.file = open(self.log_path, "a", encoding="utf-8")
        self.start_time = time.time()

    def log(self, message: str, also_print: bool = True):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        formatted = f"[{timestamp}] {message}"
        self.file.write(formatted + "\n")
        self.file.flush()
        if also_print:
            print(message)

    def log_config(self, cfg: dict, extra_info: dict = None):
        self.log("=" * 60)
        self.log("TRAINING CONFIGURATION")
        self.log("=" * 60)
        self.log(f"  excel_path:     {cfg['data']['excel_path']}")
        self.log(f"  image_root:     {cfg['data']['image_root']}")
        self.log(f"  image_size:     {cfg['data']['image_size']}")
        self.log(f"  val_ratio:      {cfg['data']['val_ratio']}")
        self.log(f"  test_ratio:     {cfg['data'].get('test_ratio', 0.0)}")
        self.log(f"  num_workers:    {cfg['data']['num_workers']}")
        if cfg["data"].get("crop"):
            c = cfg["data"]["crop"]
            self.log(f"  border_crop:    top={c.get('top',0):.2f} bottom={c.get('bottom',0):.2f} "
                     f"left={c.get('left',0):.2f} right={c.get('right',0):.2f}")
        self.log(f"  backbone:       {cfg['model']['backbone']}")
        self.log(f"  pretrained:     {cfg['model']['pretrained']}")
        self.log(f"  embed_dim:      {cfg['model']['embed_dim']}")
        self.log(f"  fusion_heads:   {cfg['model']['fusion_heads']}")
        self.log(f"  fusion_hidden:  {cfg['model'].get('fusion_hidden_dim', 128)}")
        self.log(f"  dropout:        {cfg['model']['dropout']}")
        self.log(f"  batch_size:     {cfg['train']['batch_size']}")
        self.log(f"  epochs:         {cfg['train']['epochs']}")
        self.log(f"  learning_rate:  {cfg['train']['learning_rate']}")
        self.log(f"  weight_decay:   {cfg['train']['weight_decay']}")
        self.log(f"  optimizer:      {cfg['train']['optimizer']}")
        self.log(f"  scheduler:      {cfg['train']['scheduler']}")
        self.log(f"  loss:           {cfg['loss']['name']}")
        self.log(f"  threshold:      {cfg['eval']['threshold']}")
        self.log(f"  seed:           {cfg['seed']}")
        if extra_info:
            for k, v in extra_info.items():
                self.log(f"  {k}:{' ' * (15 - len(k))}{v}")
        self.log("=" * 60)

    def log_metrics(self, epoch: int, train_loss: float, metrics: dict, prefix: str = "Val"):
        def _auc(v):
            return "N/A" if (v != v) else f"{v:.4f}"

        msg = (
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"{prefix} micro-F1: {metrics['micro_f1']:.4f} | "
            f"{prefix} macro-F1: {metrics['macro_f1']:.4f} | "
            f"{prefix} micro-AUC: {_auc(metrics.get('micro_auc', float('nan')))} | "
            f"{prefix} macro-AUC: {_auc(metrics.get('macro_auc', float('nan')))} | "
            f"{prefix} ACC: {metrics['exact_match_acc']:.4f}"
        )
        self.log(msg)

    def close(self):
        self.file.close()

    def get_elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
