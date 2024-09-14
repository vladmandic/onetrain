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
    buckets = {},
    start = None,
    progress = None, # modules.util.TrainProgress.TrainProgress
)


# args inf using functions directly
class TrainArgs():
    concept: str
    input: str
    train: bool
    validate: bool
    caption: bool
    triton: bool
    resume: bool
    bias: bool
    te: bool
    nopbar: bool
    model: str
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
    tmp: str


accelerator = accelerate.Accelerator()


def set_path(args: TrainArgs):
    if args.onetrainer:
        os.chdir(args.onetrainer)
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())


def clean_dict(d, args: TrainArgs):
    def maybe_none(v):
        if isinstance(v, float) or isinstance(v, int):
            return v > 0
        if isinstance(v, str):
            return v not in ['NONE', 'NEVER', 'EPOCH', ''] and not v.startswith('__') and not v.startswith('sample') and args.tmp not in v
        if isinstance(v, bool):
            return v
        return v

    if isinstance(d, dict):
        return dict((k, clean_dict(v, args)) for k, v in d.items() if maybe_none(k) and maybe_none(v) and clean_dict(v, args))
    elif isinstance(d, list):
        return [clean_dict(v, args) for v in d if maybe_none(v) and clean_dict(v, args)]
    else:
        return d
