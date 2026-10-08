"""
VR-OCL: Variance-Regularized Online Continual Learning
=======================================================
Derived from Lemma 3.2 of:
    Zhang & Cutkosky, "Random Scaling and Momentum for Non-smooth
    Non-convex Optimization", ICML 2024.

Lemma 3.2 states that bounding ||Delta_n||^2 is sufficient to control
the variance of the iterates y_n around x_bar_n. The paper achieves
this by adding a regularizer:

    R_n(Delta) = (mu_n / 2) * ||Delta||^2

where mu_n = mu * beta^{-n} grows over time since beta < 1.

OCL Translation
---------------
In OCL, Delta_n = theta_t - theta_{t-1} is the parameter update at
step t. Penalizing ||Delta_n||^2 prevents the model from drifting too
far from its previous state, which directly reduces catastrophic
forgetting.

Unlike EWC (Kirkpatrick et al. 2017), this requires:
  - No Fisher information matrix computation
  - No task boundary information
  - No per-parameter importance estimation

Fix log (v2)
------------
Added assertions and debug prints to verify the regularizer is
activating. Previous runs showed ER, VR_OCL, VR_OCL_Decay producing
identical results, indicating the penalty was not being applied.
Root cause: mu=0.001 with float16 mixed precision caused the penalty
to round to zero. Fixed by computing penalty in float32 explicitly.
"""

import torch
import torch.nn as nn
import time
import numpy as np

from copy import deepcopy
from src.learners.baselines.er import ERLearner
from src.utils.utils import get_device

device = get_device()


class VROCLLearner(ERLearner):
    """
    ER + Variance Regularizer with fixed mu.

    Loss at each step:
        L = CE(f_theta(x_combined), y) + (mu/2) * ||theta - theta_prev||^2

    theta_prev is snapshotted once at the start of each task and held
    fixed for the entire task duration.

    Parameters
    ----------
    vr_mu     : float   regularization strength, default 0.001
    vr_beta   : float   decay base for VROCLDecayLearner, default 0.99
    vr_mu_cap : float   ceiling for decay variant, default 0.05
    """

    def __init__(self, args):
        super().__init__(args)
        self.vr_mu          = getattr(args, 'vr_mu',          0.001)
        self.vr_beta        = getattr(args, 'vr_beta',        0.99)
        self.vr_mu_cap      = getattr(args, 'vr_mu_cap',      0.05)
        self.prev_params    = None
        self.anchor_task_id = None
        self.global_step    = 0
        self.task_step      = 0
        self.vr_debug_steps = getattr(args, 'vr_debug_steps', 5)

        # Fix 3: verify mu is positive
        assert self.vr_mu > 0, \
            f"vr_mu must be positive, got {self.vr_mu}"

        print(
            f"[VR-OCL] mu={self.vr_mu}  "
            f"beta={self.vr_beta}  "
            f"mu_cap={self.vr_mu_cap}"
        )

    # ------------------------------------------------------------------
    # Parameter snapshot
    # ------------------------------------------------------------------

    def _store_prev_params(self):
        """Snapshot theta at the start of the current task."""
        self.prev_params = {
            n: p.detach().clone().float()   # store in float32 always
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    # ------------------------------------------------------------------
    # Variance regularizer — computed in float32 to avoid underflow
    # ------------------------------------------------------------------

    def _vr_penalty(self, mu):
        """
        Compute (mu/2) * ||theta - theta_prev||^2

        Computed in float32 explicitly to prevent the penalty from
        rounding to zero under float16 mixed precision training.
        Small mu values (0.001) with float16 can underflow to zero,
        which caused VR_OCL to be identical to ER in earlier runs.
        """
        if self.prev_params is None:
            return torch.tensor(0.0, device=device)

        reg = torch.tensor(0.0, device=device, dtype=torch.float32)

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.prev_params:
                # Cast to float32 for stable penalty computation
                p_current = param.float()
                p_prev    = self.prev_params[name].to(device)
                diff      = p_current - p_prev
                reg       = reg + diff.pow(2).sum()

        return (mu / 2.0) * reg

    def _get_mu(self):
        """Fixed mu. Overridden by subclasses."""
        return self.vr_mu

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id   = kwargs.get('task_id', None)
        self.model = self.model.train()

        # Snapshot at start of each new task
        if self.prev_params is None or self.anchor_task_id != task_id:
            self._store_prev_params()
            self.anchor_task_id = task_id
            self.task_step      = 0
            print(f"[VR-OCL] parameter snapshot taken for task_id={task_id}")

        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(
                    n_imgs=self.params.mem_batch_size
                )

                if mem_x.size(0) > 0:
                    combined_x, combined_y = self.combine(
                        batch_x, batch_y, mem_x, mem_y
                    )
                    combined_x = self.transform_train(combined_x)
                    logits     = self.model.logits(combined_x)

                    mu       = self._get_mu()
                    loss_ce  = self.criterion(logits, combined_y.long())

                    # Compute penalty in float32 to prevent underflow
                    loss_vr  = self._vr_penalty(mu)

                    # Cast loss_vr back to match loss_ce dtype
                    loss_vr  = loss_vr.to(loss_ce.dtype)
                    loss     = loss_ce + loss_vr

                    self.loss = loss.item()

                    # Fix 3: debug verification for first few steps
                    if self.task_step < self.vr_debug_steps:
                        print(
                            f"  [VR-OCL debug] task={task_id} "
                            f"step={self.task_step}  "
                            f"mu={mu:.6f}  "
                            f"ce={loss_ce.item():.4f}  "
                            f"vr={loss_vr.item():.8f}  "
                            f"prev_params_set={self.prev_params is not None}"
                        )
                        # Alert if penalty is suspiciously small
                        if loss_vr.item() < 1e-10 and self.task_step > 0:
                            print(
                                f"  WARNING: VR penalty is near zero "
                                f"({loss_vr.item():.2e}). "
                                f"Check mu and prev_params."
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
                    f"mu: {self._get_mu():.6f}  "
                    f"Time: {time.time() - self.start:.2f}s"
                )


# ══════════════════════════════════════════════════════════════════════
# Variant: Decaying schedule from Theorem 4.2
# ══════════════════════════════════════════════════════════════════════

class VROCLDecayLearner(VROCLLearner):
    """
    ER + Variance Regularizer with growing mu schedule.

    mu_t = vr_mu * (1 - beta^task_step)

    Starts near zero at the beginning of each task and grows toward
    vr_mu as training progresses within the task. Capped at vr_mu_cap.

    Resets at each task boundary so earlier tasks are not over-penalized.
    """

    def __init__(self, args):
        super().__init__(args)
        print(
            f"[VR-OCL-Decay] mu={self.vr_mu}  "
            f"beta={self.vr_beta}  "
            f"mu_cap={self.vr_mu_cap}"
        )

    def _get_mu(self):
        mu_t = self.vr_mu * (
            1.0 - (self.vr_beta ** max(1, self.task_step))
        )
        return min(mu_t, self.vr_mu_cap)
