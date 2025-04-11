import io
import os
import sys
import base64
import types
import accelerate
import huggingface_hub as hf


class Obj:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


# info class that can be accessed externally
info = types.SimpleNamespace(
    id = '',
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
    images = [],
    warnings = [],
    progress = None, # modules.util.TrainProgress.TrainProgress
    validation = {}, # metadata from validation
    metadata = {}, # metadata from training
)

# args inf using functions directly
class TrainArgs():
    concept: str
    type: str
    author: str
    trigger: str

    model: str
    input: str
    output: str
    onetrainer: str
    log: str
    tmp: str

    config: str
    reference: str

    validate: bool = False
    caption: bool = False
    tag: bool = False
    train: bool = False
    sample: bool = False
    rembg: bool = False
    save: bool = False
    backup: bool = False
    resume: bool = False
    interval: int = 0

    te: bool
    nobias: bool
    gradient: bool
    epochs: int
    accumulation: int
    optimizer: str
    scheduler: str
    warmup: int
    lr: float
    d: float
    dropoout: float
    rank: int
    alpha: int
    batch: int
    resolution: int

    id: str
    format: str
    debug: bool
    triton: bool
    nopbar: bool
    noclean: bool


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
    device = accelerator.device

    if torch.cuda.is_available() and 'cuda' in str(device):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        avail, total = torch.cuda.mem_get_info()
        log.debug(f'cuda memory: avail={avail / 1024 / 1024 / 1024:.3f} total={total / 1024 / 1024 / 1024:.3f}')
    else:
        log.debug('CUDA not available or device is CPU. Skipping CUDA memory cleanup.')


def login():
    token = os.environ.get('HF_TOKEN', None)
    if token is not None:
        hf.login(token=token, add_to_git_credential=False, write_permission=False)


def b64(image):
    with io.BytesIO() as stream:
        image = image.convert('RGB')
        image.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def cache_dir():
    return os.environ.get('HF_HUB_CACHE', None)
