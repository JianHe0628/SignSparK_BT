"""Sign language back-translation: pose features to spoken language text.

    from signspark_bt import make_back_translation_model, back_translate

Training and evaluation run from the command line; see signspark_bt.cli.
"""
from .version import __version__

__all__ = ["__version__", "make_back_translation_model", "back_translate"]


def __getattr__(name):
    # Lazy so `import signspark_bt` does not pull in torch.
    if name in ("make_back_translation_model", "back_translate"):
        from . import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
