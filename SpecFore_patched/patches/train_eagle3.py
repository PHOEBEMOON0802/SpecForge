from __future__ import annotations

import argparse
import os
import sys

import torch

import specforge.utils as utils_mod

from SpecFore_patched.state import (
    is_vocab_mapping_disabled,
    set_disable_vocab_mapping,
    temporary_disable_vocab_mapping,
)

_TRAIN_MODULE = None
_ORIGINAL_BUILD_DATALOADERS = None
_ORIGINAL_GENERATE_VOCAB_MAPPING_FILE = None
_ORIGINAL_PARSE_ARGS = None
_ORIGINAL_PRINT_WITH_RANK = None


def _get_train_module():
    global _TRAIN_MODULE
    if _TRAIN_MODULE is None:
        import scripts.train_eagle3 as train_module

        _TRAIN_MODULE = train_module
    return _TRAIN_MODULE


def _patched_parse_args():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--disable-vocab-mapping",
        action="store_true",
        help="Disable draft vocab compression and train with an identity vocab mapping.",
    )
    original_argv = list(sys.argv)
    extra_args, remaining = extra_parser.parse_known_args(original_argv[1:])
    try:
        # Parse our extra flag first, then delegate the remaining argv to the
        # stock parser without rewriting the original script.
        sys.argv = [original_argv[0], *remaining]
        parser, args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.disable_vocab_mapping = extra_args.disable_vocab_mapping
    set_disable_vocab_mapping(extra_args.disable_vocab_mapping)
    return parser, args


def _load_draft_model_config(train_module, args, disable_vocab_mapping: bool):
    if args.draft_model_config is None:
        auto_config_path = train_module.create_draft_config_from_target(
            target_model_path=args.target_model_path,
            cache_dir=args.model_download_dir,
            use_vocab_mapping=not disable_vocab_mapping,
        )
        return train_module.AutoDraftModelConfig.from_file(auto_config_path)
    return train_module.AutoDraftModelConfig.from_file(args.draft_model_config)


def _resolve_checkpoint_source(train_module, args, draft_model_config):
    draft_model_last_checkpoint = None
    is_resume_checkpoint = False
    ckpt_info = (0, 0)

    if args.ckpt_dir is not None:
        if not os.path.isdir(args.ckpt_dir):
            raise ValueError(
                f"Provided base model dir {args.ckpt_dir} is not a valid directory."
            )
        draft_model_config = train_module.AutoDraftModelConfig.from_file(
            os.path.join(args.ckpt_dir, "config.json")
        )
        draft_model_last_checkpoint = args.ckpt_dir
        train_module.print_on_rank0(
            f"Finetuning from base model: {draft_model_last_checkpoint}"
        )

    if args.resume and os.path.isdir(args.output_dir):
        train_module.print_on_rank0(args.output_dir)
        draft_model_last_checkpoint, ckpt_info = train_module.get_last_checkpoint(
            args.output_dir
        )
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")
        is_resume_checkpoint = True

    return (
        draft_model_config,
        draft_model_last_checkpoint,
        is_resume_checkpoint,
        ckpt_info,
    )


def _apply_vocab_mapping_mode(
    train_module,
    draft_model_config,
    draft_model_last_checkpoint,
    disable_vocab_mapping: bool,
):
    if not disable_vocab_mapping:
        return draft_model_config

    if (
        draft_model_last_checkpoint
        and draft_model_config.draft_vocab_size != draft_model_config.vocab_size
    ):
        raise ValueError(
            "Cannot disable vocab mapping when loading a checkpoint trained "
            "with a reduced draft vocabulary."
        )

    draft_model_config.draft_vocab_size = draft_model_config.vocab_size
    train_module.print_on_rank0(
        "Vocab mapping disabled. Using full target vocabulary for the draft model."
    )
    return draft_model_config


def _build_draft_model_instance(
    train_module,
    args,
    draft_model_config,
    checkpoint_path,
):
    if checkpoint_path:
        return train_module.AutoEagle3DraftModel.from_pretrained(
            checkpoint_path,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    return train_module.AutoEagle3DraftModel.from_config(
        draft_model_config,
        attention_backend=args.attention_backend,
        torch_dtype=torch.bfloat16,
    ).cuda()


def _load_resume_state(train_module, checkpoint_path, is_resume_checkpoint):
    if not (is_resume_checkpoint and checkpoint_path):
        return None

    training_state_path = os.path.join(checkpoint_path, "training_state.pt")
    if not os.path.exists(training_state_path):
        return None

    resume_state = torch.load(
        training_state_path,
        map_location="cpu",
        weights_only=False,
    )
    train_module.print_on_rank0(
        f"Loaded training state from {training_state_path}: "
        f"epoch={resume_state['epoch']}, step={resume_state['global_step']}"
    )
    return resume_state


def _patched_build_draft_model(args):
    train_module = _get_train_module()
    disable_vocab_mapping = getattr(args, "disable_vocab_mapping", False)

    with temporary_disable_vocab_mapping(disable_vocab_mapping):
        draft_model_config = _load_draft_model_config(
            train_module,
            args,
            disable_vocab_mapping,
        )
        (
            draft_model_config,
            checkpoint_path,
            is_resume_checkpoint,
            ckpt_info,
        ) = _resolve_checkpoint_source(train_module, args, draft_model_config)
        draft_model_config = _apply_vocab_mapping_mode(
            train_module,
            draft_model_config,
            checkpoint_path,
            disable_vocab_mapping,
        )
        draft_model = _build_draft_model_instance(
            train_module,
            args,
            draft_model_config,
            checkpoint_path,
        )
        resume_state = _load_resume_state(
            train_module,
            checkpoint_path,
            is_resume_checkpoint,
        )
        draft_model.load_embedding(
            args.target_model_path,
            embedding_key=args.embedding_key,
        )
        draft_model.freeze_embedding()
        return draft_model_config, draft_model, ckpt_info, resume_state


def _patched_generate_vocab_mapping_file(*args, **kwargs):
    if is_vocab_mapping_disabled():
        return None
    return _ORIGINAL_GENERATE_VOCAB_MAPPING_FILE(*args, **kwargs)


def _patched_build_dataloaders(args, draft_model_config, processor=None):
    disable_vocab_mapping = getattr(args, "disable_vocab_mapping", False)
    with temporary_disable_vocab_mapping(disable_vocab_mapping):
        return _ORIGINAL_BUILD_DATALOADERS(args, draft_model_config, processor)


def _patched_print_with_rank(message, *args, **kwargs):
    if is_vocab_mapping_disabled() and message == "Loaded vocab mapping":
        message = "Vocab mapping disabled; using full target vocabulary."
    return _ORIGINAL_PRINT_WITH_RANK(message, *args, **kwargs)


def apply_train_eagle3_patch() -> None:
    global _ORIGINAL_BUILD_DATALOADERS
    global _ORIGINAL_GENERATE_VOCAB_MAPPING_FILE
    global _ORIGINAL_PARSE_ARGS
    global _ORIGINAL_PRINT_WITH_RANK

    if getattr(apply_train_eagle3_patch, "_applied", False):
        return

    train_module = _get_train_module()
    _ORIGINAL_PARSE_ARGS = train_module.parse_args
    _ORIGINAL_BUILD_DATALOADERS = train_module.build_dataloaders
    _ORIGINAL_GENERATE_VOCAB_MAPPING_FILE = train_module.generate_vocab_mapping_file
    _ORIGINAL_PRINT_WITH_RANK = train_module.print_with_rank

    # Patch the names imported into the script module so wrapper code can use the
    # extended signature without touching the original file.
    train_module.create_draft_config_from_target = (
        utils_mod.create_draft_config_from_target
    )

    train_module.parse_args = _patched_parse_args
    train_module.build_draft_model = _patched_build_draft_model
    train_module.generate_vocab_mapping_file = _patched_generate_vocab_mapping_file
    train_module.build_dataloaders = _patched_build_dataloaders
    train_module.print_with_rank = _patched_print_with_rank

    apply_train_eagle3_patch._applied = True
