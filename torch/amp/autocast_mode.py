import torch
import functools
import warnings
import collections
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
from torch._six import string_classes

class autocast(object):
    def __init__(self, enabled=True, dtype=torch.half):
        supported_dtype = [torch.half, torch.bfloat16]
        if dtype not in supported_dtype :
            warnings.warn("In autocast, but the target dtype is not supported. Disable the autocast.")
            warnings.warn("Supported dtype input is: torch.half, torch.bfloat16.")
            enabled = False
            dtype = torch.half
        self._enabled = enabled
        self._dtype = dtype


    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        self.prev_dtype = torch.get_autocast_dtype()
        torch.set_autocast_enabled(self._enabled)
        torch.set_autocast_dtype(self._dtype)
        torch.autocast_increment_nesting()

    def __exit__(self, *args):
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.prev)
        torch.set_autocast_dtype(self.prev_dtype)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_autocast