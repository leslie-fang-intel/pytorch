import torch
import functools
import warnings
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
from torch._six import container_abcs, string_classes


class autocast(object):
    def __init__(self, enabled=True, dtype=torch.float32):
        # support torch.float32, torch.bf16, torch.fp16, torch.int8
        #if dtype is torch.float16 and not torch.cuda.is_available():
        #    dtype = torch.float32
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



