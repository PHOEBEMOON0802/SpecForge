from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

_DISABLE_VOCAB_MAPPING = False


def set_disable_vocab_mapping(value: bool) -> None:
    global _DISABLE_VOCAB_MAPPING
    _DISABLE_VOCAB_MAPPING = bool(value)


def is_vocab_mapping_disabled() -> bool:
    return _DISABLE_VOCAB_MAPPING


def should_use_vocab_mapping(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return not is_vocab_mapping_disabled()


@contextmanager
def temporary_disable_vocab_mapping(value: bool) -> Iterator[None]:
    previous = is_vocab_mapping_disabled()
    set_disable_vocab_mapping(value)
    try:
        yield
    finally:
        set_disable_vocab_mapping(previous)
