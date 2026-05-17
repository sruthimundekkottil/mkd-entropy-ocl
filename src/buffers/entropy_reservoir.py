"""
Entropy-Guided Reservoir Buffer
================================
Drop-in replacement for Reservoir that uses softmax entropy as an
anomaly score to decide WHICH samples to retrieve from memory.

Standard reservoir:   random_retrieve() picks uniformly at random
This buffer:          entropy_retrieve() picks the samples the model
                      is MOST uncertain about (highest entropy) —
                      these are the samples most at risk of being
                      forgotten, so replaying them is most valuable.

Storage (update) is unchanged — still uses reservoir sampling so the
memory population is unbiased. Only the RETRIEVAL strategy changes.

Connection to the optimization paper (Zhang & Cutkosky, ICML 2024):
Entropy is a proxy for "how non-smooth is the loss near this sample".
Preferring high-entropy samples for replay is analogous to focusing
updates where the objective is least smooth — exactly the regime the
paper's random-scaling trick is designed to handle.
"""

import torch
import torch.nn.functional as F
import random as r
import numpy as np
import logging as lg

from src.buffers.reservoir import Reservoir
from src.utils.utils import get_device

device = get_device()


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of the softmax distribution.
    Higher entropy  →  model is more uncertain  →  higher anomaly score.

    Args:
        logits: raw logits, shape (N, C)
    Returns:
        entropy per sample, shape (N,)
    """
    probs = F.softmax(logits, dim=1)
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-8))
    return -(probs * log_probs).sum(dim=1)


class EntropyReservoir(Reservoir):
    """
    Reservoir buffer with entropy-guided retrieval.

    New argument
    ------------
    entropy_retrieval : bool
        If True  → entropy_retrieve() is used (our contribution).
        If False → falls back to standard random_retrieve().
        Default: True
    """

    def __init__(self, max_size=200, img_size=32, nb_ch=3,
                 n_classes=10, **kwargs):
        super().__init__(
            max_size=max_size,
            img_size=img_size,
            nb_ch=nb_ch,
            n_classes=n_classes,
            **kwargs
        )
        self.entropy_retrieval = kwargs.get('entropy_retrieval', True)

    # ------------------------------------------------------------------
    # Core contribution: entropy-ranked retrieval
    # ------------------------------------------------------------------

    def entropy_retrieve(self, n_imgs: int, model, transform=None):
        """
        Retrieve the n_imgs samples from the buffer that the current
        model is MOST uncertain about (highest softmax entropy).

        Falls back to random_retrieve if buffer is too small or model
        is None.

        Args:
            n_imgs   : number of samples to retrieve
            model    : current student model (used to score samples)
            transform: optional test-time transform to apply before
                       forward pass (same as self.transform_test in
                       the learner)
        Returns:
            (imgs, labels) tensors, same as random_retrieve
        """
        n_available = min(self.n_added_so_far, self.max_size)

        # Not enough data or no model → random fallback
        if n_available < n_imgs or model is None:
            lg.debug(
                f"EntropyReservoir: falling back to random retrieve "
                f"({n_available} available, {n_imgs} requested)"
            )
            return self.random_retrieve(n_imgs)

        # Score every sample currently in the buffer
        all_imgs   = self.buffer_imgs[:n_available].to(device)
        all_labels = self.buffer_labels[:n_available]

        model.eval()
        with torch.no_grad():
            imgs_input = transform(all_imgs) if transform is not None \
                         else all_imgs
            logits = model.logits(imgs_input)          # (N, C)
            entropies = softmax_entropy(logits)        # (N,)
        model.train()

        # Pick top-n_imgs by highest entropy
        _, top_idx = entropies.topk(n_imgs, largest=True, sorted=False)
        top_idx = top_idx.cpu()

        return (
            self.buffer_imgs[top_idx].clone(),
            self.buffer_labels[top_idx].clone()
        )

    # ------------------------------------------------------------------
    # Convenience: unified retrieve respects the entropy_retrieval flag
    # ------------------------------------------------------------------

    def retrieve(self, n_imgs: int, model=None, transform=None):
        """
        Unified retrieval method.
        Calls entropy_retrieve when self.entropy_retrieval is True and
        a model is provided; otherwise falls back to random_retrieve.
        """
        if self.entropy_retrieval and model is not None:
            return self.entropy_retrieve(n_imgs, model, transform)
        return self.random_retrieve(n_imgs)
