from __future__ import annotations

import functools
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import specforge.core.eagle3 as eagle3_mod
import specforge.modeling.draft.base as draft_base_mod
import specforge.modeling.draft.llama3_eagle as llama_mod
import specforge.utils as utils_mod

from SpecFore_patched.state import should_use_vocab_mapping

_ORIGINAL_GENERATE_DRAFT_MODEL_CONFIG = utils_mod.generate_draft_model_config
_ORIGINAL_LLAMA_INIT = llama_mod.LlamaForCausalLMEagle3.__init__
_ORIGINAL_LOAD_VOCAB_MAPPING = draft_base_mod.Eagle3DraftModel.load_vocab_mapping


def _remove_optional_vocab_buffers(model: nn.Module) -> None:
    model._buffers.pop("t2d", None)
    model._buffers.pop("d2t", None)


def _patched_generate_draft_model_config(
    target_model_path: str,
    template_config_path: str = None,
    cache_dir: str = None,
    use_vocab_mapping: bool = True,
):
    config_dict = _ORIGINAL_GENERATE_DRAFT_MODEL_CONFIG(
        target_model_path=target_model_path,
        template_config_path=template_config_path,
        cache_dir=cache_dir,
    )
    if not should_use_vocab_mapping(use_vocab_mapping):
        config_dict["draft_vocab_size"] = config_dict["vocab_size"]
    return config_dict


def _patched_create_draft_config_from_target(
    target_model_path: str,
    output_dir: str = None,
    template_config_path: str = None,
    cache_dir: str = None,
    use_vocab_mapping: bool = True,
):
    rank = dist.get_rank()
    if rank == 0:
        utils_mod.print_with_rank(
            "No draft model config provided, auto-generating from target model..."
        )
        config_dict = utils_mod.generate_draft_model_config(
            target_model_path,
            template_config_path,
            cache_dir,
            use_vocab_mapping=use_vocab_mapping,
        )
    dist.barrier()

    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "configs")

    model_name = target_model_path.split("/")[-1].lower()
    output_filename = f"{model_name}-eagle3-auto.json"
    output_path = os.path.join(output_dir, output_filename)

    if rank == 0:
        utils_mod.save_draft_model_config(config_dict, output_path)
        utils_mod.print_with_rank(
            f"Auto-generated draft model config saved to: {output_path}"
        )
    dist.barrier()
    return output_path


def _patched_llama_init(self, config, quant_config=None, attention_backend="sdpa"):
    _ORIGINAL_LLAMA_INIT(
        self,
        config,
        quant_config=quant_config,
        attention_backend=attention_backend,
    )
    if self.draft_vocab_size == self.vocab_size:
        # Full-vocab mode does not need draft/target remapping buffers.
        _remove_optional_vocab_buffers(self)


def _patched_load_vocab_mapping(self, file_path: str) -> None:
    if file_path is None:
        # Training still calls load_vocab_mapping unconditionally in the stock entrypoint.
        self.vocab_mapping_loaded = False
        return
    _ORIGINAL_LOAD_VOCAB_MAPPING(self, file_path)


@torch.compile(dynamic=None)
def _compute_target_p_full_vocab(target, loss_mask):
    target_head = target.float()
    position_mask = loss_mask.int()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask


def _patched_compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        if t2d is None:
            target_p, position_mask = _compute_target_p_full_vocab(
                target=target,
                loss_mask=loss_mask,
            )
        else:
            target_p, position_mask = eagle3_mod._compute_target_p(
                target=target,
                t2d=t2d,
                loss_mask=loss_mask,
            )

        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            value=1 / target_p.shape[-1],
        )
        return target_p_padded, position_mask


def _wrap_forward_with_optional_t2d(original_forward):
    @functools.wraps(original_forward)
    def wrapped(self, *args, **kwargs):
        had_t2d = hasattr(self.draft_model, "t2d")
        if not had_t2d:
            # The original forward path dereferences draft_model.t2d directly.
            self.draft_model.t2d = None
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            if not had_t2d and getattr(self.draft_model, "t2d", object()) is None:
                delattr(self.draft_model, "t2d")

    return wrapped


def apply_vocab_mapping_patch() -> None:
    if getattr(apply_vocab_mapping_patch, "_applied", False):
        return

    utils_mod.generate_draft_model_config = _patched_generate_draft_model_config
    utils_mod.create_draft_config_from_target = _patched_create_draft_config_from_target
    llama_mod.LlamaForCausalLMEagle3.__init__ = _patched_llama_init
    draft_base_mod.Eagle3DraftModel.load_vocab_mapping = _patched_load_vocab_mapping
    eagle3_mod._compute_target_p_full_vocab = _compute_target_p_full_vocab
    eagle3_mod._compute_target_p_padded = _patched_compute_target_p_padded
    eagle3_mod.OnlineEagle3Model.forward = _wrap_forward_with_optional_t2d(
        eagle3_mod.OnlineEagle3Model.forward
    )
    eagle3_mod.QwenVLOnlineEagle3Model.forward = _wrap_forward_with_optional_t2d(
        eagle3_mod.QwenVLOnlineEagle3Model.forward
    )

    apply_vocab_mapping_patch._applied = True
