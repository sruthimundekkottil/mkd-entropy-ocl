"""
ER + MKD + Entropy-Guided Replay  (ER_EMA_Entropy)
====================================================
This is the main contribution of the project.

It combines:
  1. Momentum Knowledge Distillation (MKD) from Michel et al. ICML 2024
     — EMA teacher provides soft targets to prevent forgetting

  2. Entropy-guided replay retrieval (our addition)
     — Instead of random retrieval, we pick buffered samples the model
       is currently MOST uncertain about (highest softmax entropy)

  3. Optional: Stochastic alpha scaling (our second contribution)
     — Motivated by Zhang & Cutkosky ICML 2024, the EMA update alpha
       can be randomized with an Exp(1) scalar each step, making the
       teacher update rule stochastic rather than fixed
     — Enable with --stochastic-alpha flag

The only differences from the base ER_EMALearner are:
  a) Buffer is replaced with EntropyReservoir
  b) random_retrieve() → entropy_retrieve() in the train loop
  c) Optional: update_ema() uses Exp(1)-scaled alpha
"""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import torch.cuda.amp as amp

from copy import deepcopy

from src.learners.ema.er_ema import ER_EMALearner
from src.buffers.entropy_reservoir import EntropyReservoir
from src.utils.losses import WKDLoss
from src.utils.utils import get_device

device = get_device()
scaler = amp.GradScaler()


class ER_EMA_EntropyLearner(ER_EMALearner):
    """
    ER + MKD with entropy-guided replay retrieval.

    New CLI flags handled here:
      --stochastic-alpha   : randomise EMA alpha with Exp(1) scaling
                             (connects to Zhang & Cutkosky 2024)
      --entropy-retrieval  : toggle entropy retrieval on/off
                             (default True; set False to get plain
                              ER_EMA as baseline from the same class)
    """

    def __init__(self, args):
        # These flags must exist before ER_EMALearner.__init__ runs:
        # the parent constructor calls self.update_ema(init=True), which
        # dispatches to this subclass override.
        self.entropy_retrieval = getattr(args, 'entropy_retrieval', True)
        self.stochastic_alpha = getattr(args, 'stochastic_alpha', False)

        super().__init__(args)

        # Replace the buffer only if entropy retrieval is on
        if self.entropy_retrieval:
            self.buffer = EntropyReservoir(
                max_size=self.params.mem_size,
                img_size=self.params.img_size,
                nb_ch=self.params.nb_channels,
                n_classes=self.params.n_classes,
                drop_method=self.params.drop_method,
                entropy_retrieval=True,
            )

    # ------------------------------------------------------------------
    # Optional contribution B: stochastic EMA alpha update
    # ------------------------------------------------------------------

    def update_ema(self, init=False):
        """
        Override of ER_EMALearner.update_ema().

        When self.stochastic_alpha is True, each teacher's alpha is
        scaled by s ~ Exp(1) before the EMA update, directly applying
        the random-scaling principle from Zhang & Cutkosky (2024) to
        the MKD teacher update rule.

        When False, behaviour is identical to the original.
        """
        for i, ema_model in enumerate(self.ema_models.values()):
            alpha = self.ema_alphas[i]

            # ── Stochastic alpha (our contribution B) ─────────────────
            if self.stochastic_alpha and not init:
                s = torch.distributions.Exponential(
                    torch.tensor(1.0)
                ).sample().item()
                # Keep alpha in a sensible range: (0, 0.999)
                alpha = float(np.clip(alpha * s, 1e-6, 0.999))
            # ──────────────────────────────────────────────────────────

            correction = max(
                1,
                self.stream_idx // (
                    self.params.batch_size *
                    self.params.ema_correction_step
                )
            )

            for param, ema_param in zip(
                self.model.parameters(), ema_model.parameters()
            ):
                p = deepcopy(param.data.detach())
                if init:
                    ema_param.data.copy_(p)
                else:
                    ema_param.data.mul_(1 - alpha).add_(
                        p * alpha / (1 - alpha ** correction)
                    )

    # ------------------------------------------------------------------
    # Train loop — identical to ER_EMALearner except retrieval call
    # ------------------------------------------------------------------

    def train(self, dataloader, **kwargs):
        task_name = kwargs.get("task_name", "Unknown")
        task_id = kwargs.get('task_id', None)
        self.model = self.model.train()

        for j, batch in enumerate(dataloader):
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)

            for _ in range(self.params.mem_iters):
                # Entropy scoring is inference-only. Keep it out of the
                # training autocast context used by the MKD optimization step.
                if self.entropy_retrieval:
                    mem_x, mem_y = self.buffer.entropy_retrieve(
                        n_imgs=self.params.mem_batch_size,
                        model=self.model,
                        transform=self.transform_test,
                    )
                else:
                    mem_x, mem_y = self.buffer.random_retrieve(
                        n_imgs=self.params.mem_batch_size
                    )

                if mem_x.size(0) > 0:
                    with torch.autocast(
                        device_type='cuda', dtype=torch.float16, enabled=True
                    ):
                        combined_x = torch.cat(
                            [mem_x, batch_x]
                        ).to(device)
                        combined_y = torch.cat(
                            [mem_y, batch_y]
                        ).to(device)

                        combined_aug = self.transform_train(combined_x)

                        logits_stu = self.model.logits(combined_aug)
                        logits_stu_raw = self.model.logits(combined_x)

                        # MKD distillation loss over all teachers
                        loss_dist = 0
                        for teacher in self.ema_models.values():
                            logits_tea = teacher.logits(combined_aug)
                            if self.params.no_aug:
                                loss_dist += self.wkdloss(
                                    logits_tea.detach(), logits_stu
                                )
                            else:
                                loss_dist += (
                                    self.wkdloss(
                                        logits_tea.detach(), logits_stu
                                    ) +
                                    self.wkdloss(
                                        logits_tea.detach(), logits_stu_raw
                                    )
                                ) / 2
                        loss_dist = loss_dist / len(self.ema_models)

                        loss_ce = self.criterion(
                            logits_stu, combined_y.long()
                        )
                        loss = (
                            self.params.kd_lambda * loss_dist + loss_ce
                        ).mean()

                    self.loss = loss.item()

                    with torch.autocast(
                        device_type='cuda',
                        dtype=torch.float16,
                        enabled=False
                    ):
                        scaler.scale(loss).backward()
                        scaler.step(self.optim)
                        scaler.update()

                    # EMA update (may use stochastic alpha)
                    self.update_ema()

                    if self.params.annealing:
                        self.scheduler.step()
                    self.optim.zero_grad()

                    if (
                        self.params.measure_drift >= 0 and
                        task_id > 0
                    ):
                        self.measure_drift(task_id)

                    if not self.params.no_wandb:
                        wandb.log({
                            "loss_dist": loss_dist.item(),
                            "loss": loss.item(),
                        })

                    print(
                        f"Phase: {task_name}  "
                        f"Loss:{loss.item():.3f}  "
                        f"Loss dist:{loss_dist.item():.3f}  "
                        f"batch {j}",
                        end="\r"
                    )

            self.buffer.update(imgs=batch_x, labels=batch_y)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Phase: {task_name}  "
                    f"batch {j}/{len(dataloader)}  "
                    f"Loss: {self.loss:.4f}  "
                    f"Time: {time.time() - self.start:.4f}s",
                    end="\r"
                )
                self.save(model_name=f"ckpt_{task_name}.pth")
