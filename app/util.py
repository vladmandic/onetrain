import os
import sys
import types
import accelerate


class Obj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


# info class that can be accessed externally
info = types.SimpleNamespace(
    concept = '',
    busy = False,
    complete = 0,
    samples = 0,
    step = None,
    total = None,
    epoch = None,
    buckets = {},
    start = None,
    update = None,
    status = '',
    its = None,
    mem = None,
    progress = None, # modules.util.TrainProgress.TrainProgress
    validation = {}, # metadata from validation
    metadata = {}, # metadata from training
)


# args inf using functions directly
class TrainArgs():
    concept: str
    input: str
    model: str
    format: str
    reference: str
    config: str
    train: bool
    validate: bool
    caption: bool
    triton: bool
    resume: bool
    bias: bool
    te: bool
    sample: bool
    nopbar: bool
    noclean: bool
    tag: bool
    debug: bool
    rembg: bool
    type: str
    log: str
    output: str
    onetrainer: str
    author: str
    epochs: int
    accumulation: int
    optimizer: str
    scheduler: str
    rank: int
    alpha: int
    batch: int
    resolution: int
    backup: int
    save: int
    tmp: str


accelerator = accelerate.Accelerator()


def set_path(args: TrainArgs):
    if args.onetrainer:
        os.chdir(args.onetrainer)
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())


def clean_dict(d: TrainArgs):
    def maybe_none(v):
        if isinstance(v, float) or isinstance(v, int):
            return v > 0
        if isinstance(v, str):
            return v not in ['NONE', 'NEVER', 'EPOCH', ''] and not v.startswith('__') and not v.startswith('sample') and not v.endswith('_dir') and not v.startswith('embedding') and '/' not in v
        if isinstance(v, bool):
            return v
        return v

    if isinstance(d, dict):
        return dict((k, clean_dict(v)) for k, v in d.items() if maybe_none(k) and maybe_none(v) and clean_dict(v)) # noqa: C402
    elif isinstance(d, list):
        return [clean_dict(v) for v in d if maybe_none(v) and clean_dict(v)]
    else:
        return d


def free():
    import gc
    import torch
    from .logger import log
    gc.collect()
    with torch.cuda.device(accelerator.device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    avail, total = torch.cuda.mem_get_info()
    log.debug(f'cuda memory: avail={avail / 1024 / 1024 / 1024:.3f} total={total / 1024 / 1024 / 1024:.3f}')
