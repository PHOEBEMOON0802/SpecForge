from .loss_average import apply_loss_average_patch
from .train_eagle3 import apply_train_eagle3_patch
from .vocab_mapping import apply_vocab_mapping_patch

__all__ = [
    "apply_loss_average_patch",
    "apply_train_eagle3_patch",
    "apply_vocab_mapping_patch",
]
