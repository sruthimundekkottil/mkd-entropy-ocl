"""
VR-OCL: Variance-Regularized Online Continual Learning
=======================================================
Derived from Lemma 3.2 of:
    Zhang & Cutkosky, "Random Scaling and Momentum for Non-smooth
    Non-convex Optimization", ICML 2024.

Lemma 3.2 states that bounding ||Δₙ||² is sufficient to control
the variance of the iterates yₙ around x̄ₙ. The paper achieves this
by adding a regularizer:

    Rₙ(Δ) = (μₙ/2) ||Δ||²

where μₙ = μ · β⁻ⁿ (grows over time since β < 1).

OCL Translation
---------------
In OCL, Δₙ = θₜ - θₜ₋₁ is the parameter update at step t.
Penalizing ||Δₙ||² prevents the model from drifting too far from
its previous state in a single step, which directly reduces
catastrophic forgetting — the model cannot overwrite old knowledge
too aggressively.

Unlike EWC (Kirkpatrick et al., 2017), this requires:
  - No Fisher information matrix computation
  - No task boundary information
  - No per-parameter importance estimation

It is a pure update-magnitude penalty, theoretically grounded in
the variance bound of Lemma 3.2.

Two variants are implemented:
  VR_OCL       — fixed μ throughout training
  VR_OCL_Decay — μₜ = μ · β⁻ᵗ (grows over time, as in the paper)
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
    ER + Variance Regularizer (fixed μ).

    The regularizer penalises the squared L2 norm of the parameter
    update at every gradient step:

        loss = CE(logits, y) + (μ/2) · ||θ - θ_prev||²

    Parameters
    ----------
    vr_mu : float
        Regularization strength μ. Default 0.1.
        Higher μ → less forgetting but slower plasticity.
    """

    def __init__(self, args):
        super().__init__(args)
        self.vr_mu   = getattr(args, 'vr_mu',   0.1)
        self.vr_beta = getattr(args, 'vr_beta',  0.99)  # for decay variant
        self.prev_params = None   # stores θ_{t-1}
        self.global_step = 0      # counts total gradient steps

        print(f"[VR-OCL] mu={self.vr_mu}  beta={self.vr_beta}")

    # ------------------------------------------------------------------
    # Core: variance regularizer
    # ------------------------------------------------------------------

    def _store_prev_params(self):
        """Snapshot current parameters as θ_{t-1}."""
        self.prev_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def _vr_penalty(self, mu):
        """
        Compute (μ/2) · ||θ - θ_prev||²

        This is the regularizer Rₙ(Δ) from the paper where Δ = θ - θ_prev.
        Returns scalar tensor.
        """
        if self.prev_params is None:
            return torch.tensor(0.0, device=device)

        reg = torch.tensor(0.0, device=device)
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.prev_params:
                diff = param - self.prev_params[name].to(device)
                reg  = reg + diff.pow(2).sum()

        return (mu / 2.0) * reg

    def _get_mu(self):
        """
        Return the regularization coefficient for the current step.
        Fixed μ for VROCLLearner. Overridden in VROCLDecayLearner.
        """
        return self.vr_mu

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

                    # ── Core contribution ──────────────────────────────
                    # Snapshot params BEFORE this gradient step
                    self._store_prev_params()

                    mu        = self._get_mu()
                    loss_ce   = self.criterion(logits, combined_y.long())
                    loss_vr   = self._vr_penalty(mu)
                    loss      = loss_ce + loss_vr
                    # ──────────────────────────────────────────────────

                    self.loss = loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.global_step += 1

                    if self.params.measure_drift >= 0 and task_id > 0:
                        self.measure_drift(task_id)

            self.buffer.update(imgs=batch_x, labels=batch_y,
                               model=self.model)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task: {task_name}  "
                    f"batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  "
                    f"mu: {self._get_mu():.6f}  "
                    f"Time: {time.time() - self.start:.2f}s"
                )


class VROCLDecayLearner(VROCLLearner):
    """
    ER + Variance Regularizer with decaying schedule μₜ = μ · β⁻ᵗ.

    Directly implements the schedule from Theorem 4.2 of
    Zhang & Cutkosky (2024), where μₜ = β⁻ᵗ · μ.

    Since β < 1, β⁻ᵗ grows over time, meaning the regularization
    becomes STRONGER as more tasks are seen. This means:
      - Early tasks: model can change freely (low μ)
      - Later tasks: updates are penalised more (high μ)
      - Intuition: protect accumulated knowledge more as stream grows

    Parameters
    ----------
    vr_mu   : float  — base regularization strength (default 0.01)
    vr_beta : float  — decay base β ∈ (0,1) (default 0.99)
    vr_mu_cap : float — maximum allowed μ to prevent explosion (default 10.0)
    """

    def __init__(self, args):
        super().__init__(args)
        self.vr_mu_cap = getattr(args, 'vr_mu_cap', 10.0)
        print(
            f"[VR-OCL-Decay] mu={self.vr_mu}  "
            f"beta={self.vr_beta}  "
            f"mu_cap={self.vr_mu_cap}"
        )

    def _get_mu(self):
        """
        μₜ = μ · β⁻ᵗ   (grows over time)

        Capped at vr_mu_cap to prevent numerical issues in long streams.
        """
        mu_t = self.vr_mu * (self.vr_beta ** (-self.global_step))
        return min(mu_t, self.vr_mu_cap)
