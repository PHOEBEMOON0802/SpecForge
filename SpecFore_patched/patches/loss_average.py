from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn

import specforge.core.eagle3 as eagle3_mod
import specforge.core.eagle3_adapters as adapters_mod
import specforge.core.loss as loss_mod


@torch.compile(dynamic=None)
def _patched_compute_loss(logits, target_p, position_mask):
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).sum()
    denom = position_mask.sum().clamp_min(1).to(loss.dtype)
    return loss / denom


class PatchedLogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        batch_size, seq_len, vocab_size = logits.shape
        loss = torch.zeros((batch_size * seq_len, 1), device=logits.device)
        logits_flat = logits.contiguous().view(batch_size * seq_len, vocab_size)
        target_flat = target.contiguous().view(batch_size * seq_len, vocab_size)
        position_mask_flat = position_mask.contiguous().view(
            batch_size * seq_len, 1
        ).bool()
        valid_count = max(int(position_mask_flat.sum().item()), 1)
        grid = (batch_size * seq_len,)
        m = torch.zeros(
            (batch_size * seq_len,), device=logits.device, dtype=torch.float32
        )
        d = torch.zeros(
            (batch_size * seq_len,), device=logits.device, dtype=torch.float32
        )
        block_size, num_warps = loss_mod._calculate_settings(vocab_size)
        loss_mod.log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            vocab_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        ctx.valid_count = valid_count
        ctx.save_for_backward(logits.detach(), target, position_mask, m, d)
        return loss.sum() / valid_count

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        batch_size, seq_len, vocab_size = logits.shape
        scaling_factor = 1.0 / ctx.valid_count
        logits = logits.contiguous().view(batch_size * seq_len, vocab_size)
        target = target.contiguous().view(batch_size * seq_len, vocab_size)
        position_mask = position_mask.contiguous().view(batch_size * seq_len, 1).bool()
        grid = (batch_size * seq_len,)
        block_size, num_warps = loss_mod._calculate_settings(vocab_size)
        loss_mod.log_softmax_backward_kernel[grid](
            logits,
            logits.stride(0),
            target,
            target.stride(0),
            position_mask,
            grad_output,
            scaling_factor,
            m,
            d,
            vocab_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        logits = logits.view(batch_size, seq_len, vocab_size)
        return logits, None, None, None, None


def _patched_backend_reduce_loss(
    self, loss: torch.Tensor, valid_count: torch.Tensor
) -> torch.Tensor:
    return loss


def _patched_usp_reduce_loss(
    self, loss: torch.Tensor, valid_count: torch.Tensor
) -> torch.Tensor:
    # Each rank now reports a mean over its valid positions, so we need a
    # weighted reduction to recover the global mean across sequence shards.
    weighted_loss = dist_nn.all_reduce(
        loss * valid_count,
        op=dist.ReduceOp.SUM,
        group=self.sp_group,
    )
    global_count = dist_nn.all_reduce(
        valid_count,
        op=dist.ReduceOp.SUM,
        group=self.sp_group,
    )
    return weighted_loss / global_count.clamp_min(1e-6)


def _patched_acc_and_loss(
    self,
    *,
    logits: torch.Tensor,
    target_p: torch.Tensor,
    position_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    adapter: adapters_mod.BackendAdapter,
):
    with torch.no_grad():
        local_correct = (
            (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
        ).sum()
        local_denom = loss_mask.sum().clamp_min(1e-6)
        local_correct, local_denom = adapter.reduce_metrics(
            local_correct=local_correct,
            local_denom=local_denom,
        )
        acc = local_correct / local_denom

    loss = eagle3_mod.LogSoftmaxLoss.apply(logits, target_p, position_mask)
    valid_count = position_mask.sum().to(loss.dtype).clamp_min(1.0)
    loss = adapter.reduce_loss(loss, valid_count)
    return acc, loss


def apply_loss_average_patch() -> None:
    if getattr(apply_loss_average_patch, "_applied", False):
        return

    loss_mod._compute_loss = _patched_compute_loss
    loss_mod.LogSoftmaxLoss = PatchedLogSoftmaxLoss
    eagle3_mod.LogSoftmaxLoss = PatchedLogSoftmaxLoss
    adapters_mod.BackendAdapter.reduce_loss = _patched_backend_reduce_loss
    adapters_mod.UspAdapter.reduce_loss = _patched_usp_reduce_loss
    eagle3_mod.OnlineEagle3Model._acc_and_loss = _patched_acc_and_loss

    apply_loss_average_patch._applied = True
