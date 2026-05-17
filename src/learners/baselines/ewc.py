"""
EWC: Elastic Weight Consolidation
===================================
Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural
Networks", PNAS 2017.

Implemented here as a baseline to compare against VR-OCL.

EWC penalises changes to weights that were important for previous
tasks, where importance is measured by the diagonal of the Fisher
information matrix:

    loss = CE(logits, y) + (λ/2) · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²

Where:
  θ*  = parameters after learning previous task
  F   = diagonal Fisher information (gradient² averaged over data)
  λ   = regularization strength

Key difference from VR-OCL:
  EWC  — per-parameter importance weighting (needs Fisher + task boundary)
  VR-OCL — uniform penalty on update magnitude (needs neither)
"""

import torch
import torch.nn as nn
import time
import numpy as np

from copy import deepcopy
from torch.utils.data import DataLoader
from src.learners.baselines.er import ERLearner
from src.utils.utils import get_device

device = get_device()


class EWCLearner(ERLearner):
    """
    Online EWC with experience replay.

    Fisher information is estimated at the end of each task using
    the replay buffer contents (proxy for task data, since in OCL
    we cannot store the full dataset).

    Parameters
    ----------
    ewc_lambda : float
        Regularization strength. Default 1.0.
    ewc_online : bool
        If True, accumulate Fisher across tasks (online EWC).
        If False, recompute from scratch each task boundary.
        Default True.
    """

    def __init__(self, args):
        super().__init__(args)
        self.ewc_lambda  = getattr(args, 'ewc_lambda',  1.0)
        self.ewc_online  = getattr(args, 'ewc_online',  True)

        # Stored after each task
        self.fisher      = {}   # diagonal Fisher: {param_name: tensor}
        self.optim_params = {}  # θ* after previous task: {param_name: tensor}
        self.task_count  = 0

        print(f"[EWC] lambda={self.ewc_lambda}  online={self.ewc_online}")

    # ------------------------------------------------------------------
    # Fisher estimation
    # ------------------------------------------------------------------

    def _estimate_fisher(self):
        """
        Estimate diagonal Fisher using samples from the replay buffer.
        F_i = E[ (∂ log p(y|x) / ∂θᵢ)² ]
        Approximated as mean squared gradient over buffer samples.
        """
        self.model.eval()

        # Get all buffer samples
        buf_imgs, buf_labels = self.buffer.get_all()
        if len(buf_imgs) == 0:
            self.model.train()
            return

        new_fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        batch_size = 32
        n_batches  = max(1, len(buf_imgs) // batch_size)

        for i in range(n_batches):
            imgs   = buf_imgs[i*batch_size:(i+1)*batch_size].to(device)
            labels = buf_labels[i*batch_size:(i+1)*batch_size].to(device)

            self.model.zero_grad()
            logits = self.model.logits(self.transform_test(imgs))
            loss   = nn.CrossEntropyLoss()(logits, labels.long())
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    new_fisher[n] += p.grad.detach().pow(2)

        # Average over batches
        for n in new_fisher:
            new_fisher[n] /= n_batches

        # Online EWC: accumulate Fisher across tasks
        if self.ewc_online and len(self.fisher) > 0:
            for n in new_fisher:
                self.fisher[n] = self.fisher[n] + new_fisher[n]
        else:
            self.fisher = new_fisher

        self.model.train()

    def _store_optim_params(self):
        """Store current parameters as the reference θ*."""
        self.optim_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    # ------------------------------------------------------------------
    # EWC penalty
    # ------------------------------------------------------------------

    def _ewc_penalty(self):
        """
        Compute (λ/2) · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²

        Returns 0 if no previous task has been seen yet.
        """
        if not self.optim_params:
            return torch.tensor(0.0, device=device)

        penalty = torch.tensor(0.0, device=device)
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                diff     = p - self.optim_params[n].to(device)
                penalty  = penalty + (
                    self.fisher[n].to(device) * diff.pow(2)
                ).sum()

        return (self.ewc_lambda / 2.0) * penalty

    # ------------------------------------------------------------------
    # Task boundary hook — called by main.py after each task
    # ------------------------------------------------------------------

    def after_eval(self, **kwargs):
        """
        Called at the end of each task evaluation.
        Estimates Fisher and stores optimal parameters.
        """
        super().after_eval(**kwargs)
        self._estimate_fisher()
        self._store_optim_params()
        self.task_count += 1
        print(f"[EWC] Updated Fisher after task {self.task_count}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get('task_name', 'unknown task')
        task_id   = kwargs.get('task_id', None)
        self.model = self.model.train()

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

                    loss_ce  = self.criterion(logits, combined_y.long())
                    loss_ewc = self._ewc_penalty()
                    loss     = loss_ce + loss_ewc

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
                    f"Task: {task_name}  "
                    f"batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  "
                    f"EWC penalty: {self._ewc_penalty().item():.4f}  "
                    f"Time: {time.time() - self.start:.2f}s"
                )
