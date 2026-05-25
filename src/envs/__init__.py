"""Environment package.

Concrete environments can read environment variables at import time, so avoid
eagerly importing them when callers only need shared helpers such as
``src.envs.env``.
"""

__all__ = ["AngryBirds"]


def __getattr__(name):
    if name == "AngryBirds":
        from src.envs.angry_birds import AngryBirds

        return AngryBirds
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
