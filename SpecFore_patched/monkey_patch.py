from __future__ import annotations

from SpecFore_patched.patches import (
    apply_loss_average_patch,
    apply_train_eagle3_patch,
    apply_vocab_mapping_patch,
)


def apply_patches(*, include_train_entrypoint: bool = False) -> None:
    apply_vocab_mapping_patch()
    apply_loss_average_patch()
    if include_train_entrypoint:
        apply_train_eagle3_patch()
