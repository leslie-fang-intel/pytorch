import torch
import functools
import warnings
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
from torch._six import container_abcs, string_classes


class autocast(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        torch.set_autocast_enabled(self._enabled)
        torch.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.prev)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast



