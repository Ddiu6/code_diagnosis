"""Checkpoint-aligned early stopping: ties count as no improvement."""

import math


class EarlyStopping:
    def __init__(self, patience=0):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("early_stopping_patience must be a nonnegative integer (0 disables)")
        self.patience = patience
        self.best_score = -math.inf
        self.best_epoch = 0
        self.bad_epochs = 0
        self.last_epoch = 0

    def update(self, score, epoch):
        if epoch != self.last_epoch + 1:
            raise ValueError("EarlyStopping requires consecutive epochs starting at 1")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("Checkpoint score must be finite")
        self.last_epoch = epoch
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.patience > 0 and self.bad_epochs >= self.patience
