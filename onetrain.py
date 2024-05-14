#!/usr/bin/env python

import types
import os
import sys
import time
import json
import warnings
import argparse
import tempfile
import logging
import logging.handlers
import functools
import contextlib
import cv2
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
from tqdm import tqdm


# info class that can be accessed externally
info = types.SimpleNamespace(
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


# internals
tqdm.__init__ = functools.partialmethod(tqdm.__init__, disable=True) # hide onetrainer tqdm progress bar
log = logging.getLogger(__name__)
console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
pbar = Progress(
    TextColumn('[cyan]{task.description}'),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    TextColumn('[cyan]{task.fields[text]}'),
    console=console,
    transient=False)


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


def buckets(args: TrainArgs):
    set_path(args)
    from modules.module.BaseImageCaptionModel import BaseImageCaptionModel # pylint: disable=import-error
    info.samples = BaseImageCaptionModel._BaseImageCaptionModel__get_sample_filenames(args.input) # pylint: disable=protected-access
    info.buckets = {}
    for i in info.samples:
        img = cv2.imread(i)
        resolution = f'{img.shape[0]}x{img.shape[1]}'
        if resolution not in info.buckets:
            info.buckets[resolution] = 1
        else:
            info.buckets[resolution] += 1
    log.info(f'concept: {args.concept} samples={len(info.samples)} path={args.input} buckets={info.buckets}')


def caption(args: TrainArgs):
    if not args.caption:
        return

    set_path(args)
    import torch
    import accelerate
    from modules.module.Blip2Model import Blip2Model # pylint: disable=import-error
    from modules.module.WDModel import WDModel # pylint: disable=import-error

    def caption_progress_callback(current, total):
        if not args.nopbar:
            pbar.update(task, completed=current, total=total, text=f'{current}/{total} images')

    accelerator = accelerate.Accelerator()
    log.info(f'caption: model=blip2 path={args.input}')
    model = Blip2Model(device=accelerator.device, dtype=torch.float16)
    info.busy = True
    if not args.nopbar:
        task = pbar.add_task(description="caption blip", text="", total=0)
    with pbar if not args.nopbar else contextlib.nullcontext():
        model.caption_folder(
            sample_dir=args.input,
            initial_caption='',
            caption_prefix=f'{args.concept}, ',
            caption_postfix='',
            mode='replace',
            error_callback=lambda fn: log.error(f'caption: path={fn}'),
            include_subdirectories=False,
            progress_callback=caption_progress_callback,
        )
    if not args.nopbar:
        pbar.remove_task(task)

    log.info(f'caption: model=wd14 path={args.input}')
    model = WDModel(device=accelerator.device, dtype=torch.float16)
    if not args.nopbar:
        task = pbar.add_task(description="caption wd14", text="", total=0)
    with pbar if not args.nopbar else contextlib.nullcontext():
        model.caption_folder(
            sample_dir=args.input,
            initial_caption='',
            caption_prefix='',
            caption_postfix='',
            mode='add',
            error_callback=lambda fn: log.error(f'caption: path={fn}'),
            include_subdirectories=False,
            progress_callback=caption_progress_callback,
        )
    info.busy = False
    if not args.nopbar:
        pbar.remove_task(task)
    model = None


def set_config(args: TrainArgs):
    set_path(args)
    from modules.util.config.TrainConfig import TrainConfig # pylint: disable=import-error
    import templates

    if args.config:
        with open(args.config, encoding='utf-8') as f:
            # config_json = json.load(f)
            pass
    if args.type:
        templates.config['model_type'] = "STABLE_DIFFUSION_XL_10_BASE" if args.type == 'sdxl' else 'STABLE_DIFFUSION_15'
    if args.model:
        templates.config['base_model_name'] = args.model
    if args.optimizer:
        templates.config['optimizer']['optimizer'] = args.optimizer
    if args.scheduler:
        templates.config['learning_rate_scheduler'] = args.scheduler
    if args.rank:
        templates.config['lora_rank'] = args.rank
    if args.alpha:
        templates.config['lora_alpha'] = args.alpha
    if args.batch:
        templates.config['batch_size'] = args.batch
    if args.accumulation:
        templates.config['gradient_accumulation_steps'] = args.accumulation
    if args.resolution:
        templates.config['resolution'] = str(args.resolution)
    if args.epochs:
        templates.config['epochs'] = args.epochs
    if args.triton:
        templates.config['optimizer.use_triton'] = True
    if args.resume:
        templates.config['continue_last_backup'] = True
    if args.te:
        templates.config['text_encoder']['train'] = True
    if args.bias:
        templates.config['optimizer']['use_bias_correction'] = True
    if args.backup:
        templates.config['backup_after'] = int(templates.config['epochs'] / args.backup)
        templates.config['backup_after_unit'] = 'EPOCH'
    if args.save:
        templates.config['save_after'] = int(templates.config['epochs'] / args.save)
        templates.config['save_after_unit'] = 'EPOCH'
    templates.config['debug_dir'] = os.path.join(args.tmp, 'debug')
    templates.config['workspace_dir'] = os.path.join(args.tmp, 'workspace')
    templates.config['cache_dir'] = os.path.join(args.tmp, 'cache')
    templates.config['concept_file_name'] = os.path.join(args.tmp, 'concept.json')
    templates.config['output_model_destination'] = args.output or os.path.join(args.tmp, f'{args.concept}.safetensors')
    templates.config['sample_definition_file_name'] = os.path.join(args.tmp, 'samples.json')

    config = TrainConfig.default_values()
    config.from_dict(templates.config)

    with open(os.path.join(args.tmp, 'config.json'), "w", encoding='utf-8') as f:
        log.info(f'config: file={os.path.join(args.tmp, "config.json")}')
        json.dump(templates.config, f, indent=2)
    with open(config.concept_file_name, "w", encoding='utf-8') as f:
        templates.concepts[0]["name"] = args.concept
        templates.concepts[0]["path"] = args.input
        templates.concepts[0]["text"]["prompt_path"] = args.input
        if args.resolution:
            templates.concepts[0]["image"]["enable_resolution_override"] = True
            templates.concepts[0]["image"]["resolution_override"] = str(config.resolution)
        log.info(f'concepts: name={args.concept} file={config.concept_file_name}')
        json.dump(templates.concepts, f, indent=2)
    with open(config.sample_definition_file_name, "w", encoding='utf-8') as f:
        log.info(f'samples: file={config.sample_definition_file_name}')
        json.dump(templates.samples, f, indent=2)

    return config, templates.config


def train(args: TrainArgs):
    if not args.train:
        return

    set_path(args)
    buckets(args)

    from modules.util.callbacks.TrainCallbacks import TrainCallbacks # pylint: disable=import-error
    from modules.util.commands.TrainCommands import TrainCommands # pylint: disable=import-error
    from modules.trainer.GenericTrainer import GenericTrainer # pylint: disable=import-error

    def train_progress_callback(p, max_sample, max_epoch):
        ts = time.time()
        info.progress = p
        total = max_sample * max_epoch
        info.complete = int(100 * p.global_step / total)
        its = p.global_step / (ts - info.start)
        if not args.nopbar:
            pbar.update(task, completed=info.complete, description="train", text=f'step: {p.global_step} epoch: {p.epoch+1}/{max_epoch} batch: {p.epoch_step} samples: {max_sample} its: {its:.2f}')

    def log_update(s: str):
        if s not in ['training', 'starting epoch/caching']:
            log.info(f'update: {s}')

    info.busy = True
    callbacks = TrainCallbacks()
    callbacks.set_on_update_status(log_update)
    callbacks.set_on_update_train_progress(train_progress_callback)
    commands = TrainCommands()
    config, config_json = set_config(args)
    log.info(f'method={config.training_method} type={config.model_type}')
    log.info(f'model={config.base_model_name}')

    trainer = GenericTrainer(config, callbacks, commands)
    if not args.nopbar:
        task = pbar.add_task(description="train", text="", total=100)
    info.start = time.time()

    try:
        log.info('train: init')
        trainer.start()
        del trainer.model.model_spec.thumbnail
        trainer.model.model_spec.author = args.author
        trainer.model.model_spec.date = 'today'
        trainer.model.model_spec.config = json.dumps(clean_dict(config_json, args))
        trainer.model.model_spec.concepts = json.dumps({
            "name": args.concept,
            "images": len(info.samples),
            "buckets": info.buckets,
        })
        log.info(f'optimizer={config.optimizer.optimizer} scheduler={config.learning_rate_scheduler} rank={config.lora_rank} alpha={config.lora_alpha} batch={config.batch_size} accumulation={config.gradient_accumulation_steps} epochs={config.epochs}')
        log.info('train: start')
        with pbar if not args.nopbar else contextlib.nullcontext():
            time.sleep(1)
            trainer.train()
        trainer.end()
        log.info(f'save: {trainer.config.output_model_destination}')
        log.info('train: completed')
    except Exception as e:
        log.error(f'train: error={e}')

    info.busy = False
    if not args.nopbar:
        pbar.remove_task(task)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description = 'onetrain')
    parser.add_argument('--concept', required=True, type=str, help='concept name')
    parser.add_argument('--input', required=True, type=str, help='folder with training dataset')
    parser.add_argument("--model", required=False, type=str, help='stable diffusion base model')
    parser.add_argument("--train", default=False, action='store_true', help='run training')
    parser.add_argument("--caption", default=False, action='store_true', help='run captioning')
    parser.add_argument("--triton", default=False, action='store_true', help='use triton')
    parser.add_argument("--resume", default=False, action='store_true', help='resume training from last backup')
    parser.add_argument("--te", default=False, action='store_true', help='train text encoder')
    parser.add_argument("--bias", default=False, action='store_true', help='use bias correction')
    parser.add_argument("--config", required=False, type=str, help='use specific onetrainer config file')
    parser.add_argument("--type", required=False, choices=['sd', 'sdxl'], help='model type')
    parser.add_argument('--log', required=False, type=str, help='specify log file')
    parser.add_argument('--output', required=False, type=str, help='specify output location')
    parser.add_argument('--onetrainer', required=False, type=str, help='path to onetrainer')
    parser.add_argument('--author', required=False, type=str, help='train author to be included in metadata')
    parser.add_argument('--epochs', required=False, type=int, help='number of training epochs')
    parser.add_argument('--accumulation', required=False, type=int, help='gradient accumulation steps')
    parser.add_argument('--optimizer', required=False, type=str, help='training optiomizer')
    parser.add_argument('--scheduler', required=False, type=str, help='training scheduler')
    parser.add_argument('--rank', required=False, type=int, help='lora rank')
    parser.add_argument('--alpha', required=False, type=int, help='lora alpha')
    parser.add_argument('--batch', required=False, type=int, help='training batch size')
    parser.add_argument('--backup', required=False, type=int, help='create n training backups')
    parser.add_argument('--save', required=False, type=int, help='save n intermittent steps')
    parser.add_argument('--resolution', required=False, type=int, help='training override resolution')
    parser.add_argument("--nopbar", default=False, action='store_true', help='disable progress bar')
    parser.add_argument('--tmp', default=os.path.join(tempfile.gettempdir(), 'onetrain'), type=str, help='training temp folder')
    parsed = parser.parse_args()

    os.makedirs(parsed.tmp, exist_ok=True)
    log_handler_console = RichHandler(show_time=True, omit_repeated_times=False, show_level=True, show_path=False, markup=False, rich_tracebacks=True, log_time_format='%H:%M:%S-%f', level=logging.INFO, console=console)
    log_handler_console.setLevel(logging.INFO)
    log.addHandler(log_handler_console)
    log_file = parsed.log or os.path.join(parsed.tmp, 'onetrain.log')
    log_handler_file = logging.handlers.RotatingFileHandler(log_file, encoding='utf-8', delay=True)
    log_handler_file.formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(module)s | %(message)s')
    log_handler_file.setLevel(logging.INFO)
    log.addHandler(log_handler_file)
    log.info('onetrain')
    log.info(f'log: {log_file}')
    log.info(f'args: {parsed}')

    caption(parsed)
    train(parsed)
