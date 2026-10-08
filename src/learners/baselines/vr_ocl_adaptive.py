"""
VR-OCL Adaptive: Variance Regularizer with Adaptive mu
=======================================================
Extends VROCLLearner by computing mu automatically from
an accuracy-based gap signal rather than a loss-based gap.

Fix log
-------
v1 (original): used loss gap CE(memory) - CE(stream)
    Problem: gap goes negative in later tasks because new classes
    have higher CE loss due to difficulty, not forgetting.
    Result: mu drops to near zero, regularizer turns off.

v2 (this file): uses accuracy gap + forgetting signal
    accuracy gap  = acc(stream) - acc(memory)
    forgetting    = max(0, best_mem_acc - current_mem_acc)
    combined gap  = forgetting + max(0, str_acc - mem_acc - margin)
    This is robust to task difficulty because accuracy on old classes
    directly measures retention regardless of new class hardness.
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
    ER + Variance Regularizer with Adaptive mu (v2, accuracy gap).

    mu is computed each step from:

        forgetting    = max(0, best_mem_acc_this_task - current_mem_acc)
        acc_gap       = max(0, str_acc - mem_acc - margin)
        combined_gap  = forgetting + acc_gap
        gap_ema       = ema_beta * gap_ema + (1 - ema_beta) * combined_gap
        mu_t          = mu_max * sigmoid(gap_ema / tau)

    When memory accuracy drops from its peak -> forgetting detected
    -> gap increases -> mu increases -> regularizer tightens.

    When memory accuracy is stable -> gap near zero -> mu stays
    at mu_max * 0.5 (neutral point of sigmoid).

    Parameters
    ----------
    vr_mu_max   : float  maximum mu, default 0.05
    vr_tau      : float  sigmoid temperature, default 0.3
                         lower = more sensitive to gap changes
    vr_ema_beta : float  EMA smoothing for gap, default 0.9
    vr_margin   : float  minimum acc advantage before gap fires, default 0.05
                         prevents tiny fluctuations from changing mu
    """

    def __init__(self, args):
        super().__init__(args)
        self.vr_mu_max   = getattr(args, 'vr_mu_max',   0.05)
        self.vr_tau      = getattr(args, 'vr_tau',      0.3)
        self.vr_ema_beta = getattr(args, 'vr_ema_beta', 0.9)
        self.vr_margin   = getattr(args, 'vr_margin',   0.05)

        self.gap_ema          = 0.0
        self.gap_history      = []
        self.best_mem_acc     = 0.0   # best memory acc seen in current task
        self.current_task_id  = -1    # tracks task for reset

        print(
            f"[VR-OCL-Adaptive v2] "
            f"mu_max={self.vr_mu_max}  "
            f"tau={self.vr_tau}  "
            f"ema_beta={self.vr_ema_beta}  "
            f"margin={self.vr_margin}"
        )

    # ------------------------------------------------------------------
    # Reset best_mem_acc at each new task boundary
    # ------------------------------------------------------------------

    def _maybe_reset_task_tracking(self, task_id):
        """
        Reset best_mem_acc when a new task starts.
        This ensures the forgetting signal is relative to the current
        task's peak, not a global peak from earlier tasks.
        """
        if task_id != self.current_task_id:
            self.best_mem_acc    = 0.0
            self.current_task_id = task_id

    # ------------------------------------------------------------------
    # Accuracy-based gap (Fix 1 + Fix 2)
    # ------------------------------------------------------------------

    def _compute_gap(self, batch_x, batch_y, mem_x, mem_y):
        """
        Compute combined forgetting + accuracy gap signal.

        Uses accuracy rather than loss so the signal is not affected
        by class difficulty (new classes always have high loss even
        when the model is learning them correctly).

        Returns positive value when forgetting is detected.
        Returns near-zero when memory is being retained well.
        """
        # Fall back gracefully if buffer is too small
        if mem_x.size(0) < 10:
            return self._compute_normalised_loss_gap(
                batch_x, batch_y, mem_x, mem_y
            )

        self.model.eval()
        with torch.no_grad():
            # Memory accuracy
            mem_logits = self.model.logits(
                self.transform_test(mem_x.to(device))
            )
            mem_preds  = mem_logits.argmax(dim=1)
            mem_acc    = (
                mem_preds == mem_y.to(device)
            ).float().mean().item()

            # Stream accuracy
            str_logits = self.model.logits(
                self.transform_test(batch_x.to(device))
            )
            str_preds  = str_logits.argmax(dim=1)
            str_acc    = (
                str_preds == batch_y.to(device)
            ).float().mean().item()

        self.model.train()

        # Update best memory accuracy for this task (Fix 4)
        if mem_acc > self.best_mem_acc:
            self.best_mem_acc = mem_acc

        # Forgetting signal: how much has memory acc dropped from peak
        # Only fires when there is genuine degradation
        forgetting_signal = max(0.0, self.best_mem_acc - mem_acc)

        # Accuracy gap: stream doing much better than memory
        # margin prevents tiny fluctuations from triggering
        acc_gap = max(0.0, str_acc - mem_acc - self.vr_margin)

        # Combined signal
        combined_gap = forgetting_signal + acc_gap

        return combined_gap

    def _compute_normalised_loss_gap(self, batch_x, batch_y, mem_x, mem_y):
        """
        Fallback for when memory batch is too small for reliable accuracy.
        Uses normalised loss gap instead of raw loss gap (Fix 5).
        """
        if not hasattr(self, '_mem_loss_ema'):
            self._mem_loss_ema = 1.0
            self._str_loss_ema = 1.0

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

        alpha = 0.01
        self._mem_loss_ema = (1-alpha) * self._mem_loss_ema + alpha * loss_mem
        self._str_loss_ema = (1-alpha) * self._str_loss_ema + alpha * loss_str

        return (loss_mem / (self._mem_loss_ema + 1e-8)) - \
               (loss_str / (self._str_loss_ema + 1e-8))

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_gap_ema(self, gap):
        self.gap_ema = (
            self.vr_ema_beta       * self.gap_ema +
            (1 - self.vr_ema_beta) * gap
        )

    # ------------------------------------------------------------------
    # Adaptive mu via sigmoid
    # ------------------------------------------------------------------

    def _get_mu(self):
        """
        mu_t = mu_max * sigmoid(gap_ema / tau)

        gap_ema = 0   -> sigmoid = 0.5 -> mu = mu_max/2 (neutral)
        gap_ema >> 0  -> sigmoid -> 1  -> mu -> mu_max  (high forgetting)
        gap_ema << 0  -> sigmoid -> 0  -> mu -> 0       (no forgetting)
        """
        sigmoid_val = 1.0 / (1.0 + np.exp(-self.gap_ema / self.vr_tau))
        return self.vr_mu_max * sigmoid_val

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id   = kwargs.get('task_id', None)
        self.model = self.model.train()

        # Snapshot parameters at start of each task
        if self.prev_params is None or self.anchor_task_id != task_id:
            self._store_prev_params()
            self.anchor_task_id = task_id
            self.task_step = 0
            self._maybe_reset_task_tracking(task_id)
            print(f"[VR-OCL-Adaptive] snapshot + reset for task_id={task_id}")

        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(
                    n_imgs=self.params.mem_batch_size
                )

                if mem_x.size(0) > 0:

                    # Compute adaptive mu
                    if self.buffer.n_added_so_far >= self.params.mem_batch_size:
                        gap = self._compute_gap(
                            batch_x, batch_y, mem_x, mem_y
                        )
                        self._update_gap_ema(gap)
                        self.gap_history.append({
                            'step':       self.global_step,
                            'task_id':    task_id,
                            'gap':        gap,
                            'gap_ema':    self.gap_ema,
                            'mu':         self._get_mu(),
                            'best_mem':   self.best_mem_acc,
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
                            f"  [adaptive v2] task={task_id} "
                            f"step={self.task_step} "
                            f"gap={self.gap_ema:.4f}  "
                            f"mu={mu:.6f}  "
                            f"best_mem={self.best_mem_acc:.3f}  "
                            f"ce={loss_ce.item():.4f}  "
                            f"vr={loss_vr.item():.6f}"
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
                    f"best_mem: {self.best_mem_acc:.3f}  "
                    f"Time: {time.time() - self.start:.2f}s"
                )
