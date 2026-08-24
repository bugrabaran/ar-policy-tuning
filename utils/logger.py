import logging
import torch.distributed as dist
import torch
import time

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class PhaseTimer:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.reset()
    def reset(self):
        self.t = {}
        self._last = None
        self._last_name = None
    def tic(self, name):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._last = time.perf_counter()
        self._last_name = name
    def toc(self, name=None):
        if self.use_cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        nm = name if name is not None else self._last_name
        self.t[nm] = self.t.get(nm, 0.0) + (now - self._last)
        self._last = None
        self._last_name = None
    def get(self):
        return self.t.copy()