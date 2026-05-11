from __future__ import annotations

from SpecFore_patched.monkey_patch import apply_patches


def main() -> None:
    apply_patches(include_train_entrypoint=True)

    import scripts.train_eagle3 as train_module

    train_module.main()


if __name__ == "__main__":
    main()
