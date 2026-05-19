"""
VR-OCL Adaptive: Variance Regularizer with adaptive mu
=======================================================
Extends VROCLLearner by computing mu automatically from
the replay-stream loss gap at each training step.

See vr_ocl.py for the base class and theoretical motivation.
"""

import torch
import numpy as np
import time
import pandas as pd
import os

from src.learners.baselines.vr_ocl import VROCLLearner
from src.utils.utils import get_device

device = get_device()


class VROCLAdaptiveLearner(VROCLLearner):
    """
    ER + Variance Regularizer with Adaptive mu.

    mu is computed automatically each step from the gap between
    replay loss and stream loss:

        gap_t = CE(model, memory_batch) - CE(model, stream_batch)
        mu_t  = mu_max * sigmoid(gap_ema / tau)

    When gap > 0: model is worse on old data -> forgetting detected
                  -> sigmoid > 0.5 -> mu increases
    When gap < 0: model handles old data well -> replay sufficient
                  -> sigmoid < 0.5 -> mu decreases

    Parameters
    ----------
    vr_mu_max   : float  maximum mu (default 0.05)
    vr_tau      : float  sigmoid temperature (default 0.5)
    vr_ema_beta : float  EMA smoothing for gap (default 0.9)
    """

    def __init__(self, args):
        super().__init__(args)
        self.vr_mu_max   = getattr(args, 'vr_mu_max',   0.05)
        self.vr_tau      = getattr(args, 'vr_tau',      0.5)
        self.vr_ema_beta = getattr(args, 'vr_ema_beta', 0.9)
        self.vr_debug_steps = getattr(args, 'vr_debug_steps', 5)

        self.gap_ema     = 0.0
        self.gap_history = []

        print(
            f"[VR-OCL-Adaptive] mu_max={self.vr_mu_max}  "
            f"tau={self.vr_tau}  ema_beta={self.vr_ema_beta}"
        )

    def _compute_gap(self, batch_x, batch_y, mem_x, mem_y):
        """
        Compute loss gap between memory and stream samples.
        No gradients flow through this — it only sets mu.
        """
        self.model.eval()
        with torch.no_grad():
            mem_logits = self.model.logits(
                self.transform_test(mem_x.to(device))
            )
            loss_mem = self.criterion(
                mem_logits, mem_y.to(device).long()
            ).item()

            str_logits = self.model.logits(
                self.transform_test(batch_x.to(device))
            )
            loss_str = self.criterion(
                str_logits, batch_y.to(device).long()
            ).item()

        self.model.train()
        return loss_mem - loss_str

    def _update_gap_ema(self, gap):
        self.gap_ema = (
            self.vr_ema_beta       * self.gap_ema +
            (1 - self.vr_ema_beta) * gap
        )

    def _get_mu(self):
        sigmoid_val = 1.0 / (1.0 + np.exp(-self.gap_ema / self.vr_tau))
        return self.vr_mu_max * sigmoid_val

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id   = kwargs.get('task_id', None)
        self.model = self.model.train()

        if self.prev_params is None or self.anchor_task_id != task_id:
            self._store_prev_params()
            self.anchor_task_id = task_id
            self.task_step = 0
            print(f"[VR-OCL-Adaptive] snapshot taken for task_id={task_id}")

        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(
                    n_imgs=self.params.mem_batch_size
                )

                if mem_x.size(0) > 0:

                    # Compute adaptive mu from gap
                    if self.buffer.n_added_so_far >= self.params.mem_batch_size:
                        gap = self._compute_gap(
                            batch_x, batch_y, mem_x, mem_y
                        )
                        self._update_gap_ema(gap)
                        self.gap_history.append({
                            'step':    self.global_step,
                            'gap':     gap,
                            'gap_ema': self.gap_ema,
                            'mu':      self._get_mu(),
                        })

                    combined_x, combined_y = self.combine(
                        batch_x, batch_y, mem_x, mem_y
                    )
                    combined_x = self.transform_train(combined_x)
                    logits     = self.model.logits(combined_x)

                    mu      = self._get_mu()
                    loss_ce = self.criterion(logits, combined_y.long())
                    loss_vr = self._vr_penalty(mu)
                    loss    = loss_ce + loss_vr

                    self.loss = loss.item()

                    if self.task_step < self.vr_debug_steps:
                        print(
                            f"  [adaptive] task={task_id} "
                            f"step={self.task_step} "
                            f"gap={self.gap_ema:.4f}  "
                            f"mu={mu:.6f}  "
                            f"ce={loss_ce.item():.4f}"
                        )

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.global_step += 1
                    self.task_step   += 1

                    if self.params.measure_drift >= 0 and task_id > 0:
                        self.measure_drift(task_id)

            self.buffer.update(
                imgs=batch_x, labels=batch_y, model=self.model
            )

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task: {task_name}  "
                    f"batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  "
                    f"gap_ema: {self.gap_ema:.4f}  "
                    f"mu: {self._get_mu():.6f}  "
                    f"Time: {time.time() - self.start:.2f}s"
                )
