# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit element masks for native OpenPI SFT and accumulation."""

from typing import Any

import torch


def validate_action_loss_mask(mask: Any, actions: torch.Tensor) -> torch.Tensor:
    """Require a Boolean [B,H,D] mask with at least one supervised element."""
    mask = torch.as_tensor(mask, device=actions.device)
    if mask.dtype != torch.bool:
        raise ValueError(f"Action loss mask must be Boolean, found {mask.dtype}")
    if actions.ndim != 3 or mask.shape != actions.shape:
        raise ValueError(
            f"Action loss mask must match [B,H,D] {tuple(actions.shape)}, found {tuple(mask.shape)}"
        )
    if not mask.any().item():
        raise ValueError("Action loss mask has zero valid supervised targets")
    return mask


def masked_action_loss(
    elementwise_loss: torch.Tensor,
    mask: torch.Tensor,
    normalizer: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Sum valid losses over their count or a worker's global-step normalizer."""
    mask = validate_action_loss_mask(mask, elementwise_loss)
    selected = elementwise_loss.masked_select(mask).float()
    if not torch.isfinite(selected).all().item():
        raise ValueError("Valid supervised losses must be finite")
    denominator = (
        mask.sum().to(dtype=torch.float32)
        if normalizer is None
        else torch.as_tensor(normalizer, device=selected.device, dtype=torch.float32)
    )
    if (
        denominator.ndim != 0
        or denominator.requires_grad
        or not torch.isfinite(denominator).item()
        or denominator.item() <= 0
    ):
        raise ValueError(
            "Action loss normalizer must be a finite positive detached scalar"
        )
    return selected.sum() / denominator


def masked_micro_batch_count(batch: dict) -> int:
    """Count a collated microbatch's supervision before the forward pass."""
    if "action_loss_mask" not in batch or "actions" not in batch:
        raise ValueError("Masked SFT microbatch requires actions and action_loss_mask")
    actions = torch.as_tensor(batch["actions"])
    return int(
        validate_action_loss_mask(batch["action_loss_mask"], actions).sum().item()
    )


def normalize_masked_micro_batches(
    batches: list[dict], *, world_size: int, global_valid_count: int
) -> list[dict]:
    """Account for base-worker /G and FSDP /W without averaging local means.

    Each model returns N_micro / (D_global / (W*G)). The base worker divides
    by G and FSDP averages by W, yielding sum(N_micro) / D_global.
    """
    if not batches or type(world_size) is not int or world_size < 1:
        raise ValueError(
            "Expected nonempty accumulation batches and positive world size"
        )
    local_count = sum(masked_micro_batch_count(b) for b in batches)
    if type(global_valid_count) is not int or global_valid_count < local_count:
        raise ValueError(
            "Global valid count must include all local supervised elements"
        )
    denominator = global_valid_count / (world_size * len(batches))
    result = []
    for batch in batches:
        if "action_loss_normalizer" in batch:
            raise ValueError(
                "Action loss normalizer is owned by the accumulation worker"
            )
        result.append({**batch, "action_loss_normalizer": denominator})
    return result
