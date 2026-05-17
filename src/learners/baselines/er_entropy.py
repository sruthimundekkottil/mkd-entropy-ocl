"""
ER with Entropy-Guided Replay  (ER_Entropy)
============================================
Identical to the standard ER baseline EXCEPT that replay retrieval
uses entropy-ranked selection instead of random sampling.

The buffer is still populated via reservoir sampling (unbiased).
Only retrieval changes: we ask "which buffered samples is the model
currently most uncertain about?" and replay those.

This is our Contribution A — requires no MKD, no teacher.
"""

import torch
import time
import torch.nn as nn
import numpy as np
import wandb

from src.learners.baselines.er import ERLearner
from src.buffers.entropy_reservoir import EntropyReservoir
from src.utils.utils import get_device

device = get_device()


class ER_EntropyLearner(ERLearner):
    """ER with entropy-guided replay buffer."""

    def __init__(self, args):
        super().__init__(args)
        # Replace the standard Reservoir with our EntropyReservoir
        self.buffer = EntropyReservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
            entropy_retrieval=True,          # ← our contribution ON
        )

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id   = kwargs.get('task_id', None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                # ── KEY CHANGE: entropy-ranked retrieval ──────────────
                mem_x, mem_y = self.buffer.hybrid_retrieve(
                    n_imgs=self.params.mem_batch_size,
                    model=self.model,
                    transform=self.transform_test,
                    random_ratio=0.5,
                )
                # ─────────────────────────────────────────────────────

                if mem_x.size(0) > 0:
                    combined_x, combined_y = self.combine(
                        batch_x, batch_y, mem_x, mem_y
                    )
                    combined_x = self.transform_train(combined_x)
                    logits     = self.model.logits(combined_x)
                    loss       = self.criterion(logits, combined_y.long())

                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    if self.params.measure_drift >= 0 and task_id > 0:
                        self.measure_drift(task_id)

            self.buffer.update(imgs=batch_x, labels=batch_y,
                               model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task: {task_name}  batch {j}/{len(dataloader)}"
                    f"  Loss: {self.loss:.4f}"
                    f"  Time: {time.time() - self.start:.2f}s"
                )
