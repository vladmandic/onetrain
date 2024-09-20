import os
import time
import json
import contextlib
import datetime
import cv2
from app.logger import log
from app.util import TrainArgs, set_path, clean_dict, info
from app.config import get_config
from app.caption import tags


def set_config(args: TrainArgs):
    from modules.util.config.TrainConfig import TrainConfig # pylint: disable=import-error
    config = get_config('onetrainer')

    if args.type:
        if args.type == 'sdxl':
            config['model_type'] = "STABLE_DIFFUSION_XL_10_BASE"
        elif args.type == 'sd':
            config['model_type'] = 'STABLE_DIFFUSION_15'
        elif args.type == 'flux':
            config['model_type'] = 'FLUX_DEV_1'
        else:
            log.warning(f'Unknown Model type: {args.type}')
    if args.model:
        config['base_model_name'] = args.model
    if args.optimizer:
        config['optimizer']['optimizer'] = args.optimizer
    if args.scheduler:
        config['learning_rate_scheduler'] = args.scheduler
    if args.rank:
        config['lora_rank'] = args.rank
    if args.alpha:
        config['lora_alpha'] = args.alpha
    if args.batch:
        config['batch_size'] = args.batch
    if args.accumulation:
        config['gradient_accumulation_steps'] = args.accumulation
    if args.resolution:
        config['resolution'] = str(args.resolution)
    if args.epochs:
        config['epochs'] = args.epochs
    if args.triton:
        config['optimizer.use_triton'] = True
    if args.resume:
        config['continue_last_backup'] = True
    if args.te:
        config['text_encoder']['train'] = True
    if args.bias:
        config['optimizer']['use_bias_correction'] = True
    if args.backup:
        config['backup_after'] = int(config['epochs'] / args.backup)
        config['backup_after_unit'] = 'EPOCH'
    if args.save:
        config['save_after'] = int(config['epochs'] / args.save)
        config['save_after_unit'] = 'EPOCH'
    config['debug_dir'] = os.path.join(args.tmp, 'debug')
    config['workspace_dir'] = os.path.join(args.tmp, 'workspace')
    config['cache_dir'] = os.path.join(args.tmp, 'cache')
    config['concept_file_name'] = os.path.join(args.tmp, 'concept.json')
    config['output_model_destination'] = args.output or os.path.join(args.tmp, f'{args.concept}.safetensors')
    config['sample_definition_file_name'] = os.path.join(args.tmp, 'samples.json')

    train_config = TrainConfig.default_values()
    train_config.from_dict(config)

    with open(os.path.join(args.tmp, 'config.json'), "w", encoding='utf-8') as f:
        log.info(f'write config: file="{os.path.join(args.tmp, "config.json")}"')
        json.dump(config, f, indent=2)

    with open(train_config.concept_file_name, "w", encoding='utf-8') as f:
        concepts = get_config('concepts')
        concepts[0]["name"] = args.concept
        concepts[0]["path"] = args.input
        concepts[0]["text"]["prompt_path"] = args.input
        if args.resolution:
            concepts[0]["image"]["enable_resolution_override"] = True
            concepts[0]["image"]["resolution_override"] = str(train_config.resolution)
        log.info(f'write concepts: file="{train_config.concept_file_name}" name="{args.concept}"')
        json.dump(concepts, f, indent=2)

    with open(train_config.sample_definition_file_name, "w", encoding='utf-8') as f:
        samples = get_config('samples')
        log.info(f'write samples: file="{train_config.sample_definition_file_name}"')
        json.dump(samples, f, indent=2)

    return train_config, config


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


def train(args: TrainArgs):
    if not args.train:
        return

    info.status = 'start'
    set_path(args)
    buckets(args)

    from app.logger import pbar
    from modules.util.callbacks.TrainCallbacks import TrainCallbacks # pylint: disable=import-error
    from modules.util.commands.TrainCommands import TrainCommands # pylint: disable=import-error
    from modules.trainer.GenericTrainer import GenericTrainer # pylint: disable=import-error

    def train_progress_callback(p, max_sample, max_epoch):
        ts = time.time()
        info.progress = p
        total = max_sample * max_epoch
        info.complete = int(100 * p.global_step / total)
        info.epoch = p.epoch
        info.step = p.global_step
        info.update = time.time()
        info.status = 'train'
        its = p.global_step / (ts - info.start)
        if not args.nopbar:
            pbar.update(task, completed=info.complete, description="train", text=f'step: {p.global_step} epoch: {p.epoch+1}/{max_epoch} batch: {p.epoch_step} samples: {max_sample} its: {its:.2f}')

    def log_update(s: str):
        if 'loading' in s:
            info.status = 'loading'
        if s not in ['training', 'starting epoch/caching']:
            log.info(f'update: {s}')

    info.busy = True
    info.concept = args.concept
    callbacks = TrainCallbacks()
    callbacks.set_on_update_status(log_update)
    callbacks.set_on_update_train_progress(train_progress_callback)
    commands = TrainCommands()
    config, config_json = set_config(args)
    log.info(f'method={config.training_method} type={config.model_type}')
    log.info(f'model="{config.base_model_name}"')

    trainer = GenericTrainer(config, callbacks, commands)
    if not args.nopbar:
        task = pbar.add_task(description="train", text="", total=100)
    info.start = time.time()
    info.status = 'starting'

    try:
        log.info('train: init')
        trainer.start()
        del trainer.model.model_spec.thumbnail
        trainer.model.model_spec.author = args.author
        trainer.model.model_spec.title = args.concept
        trainer.model.model_spec.date = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        config_metadata = json.dumps(clean_dict(config_json))
        trainer.model.model_spec.base_model = os.path.basename(config_json.get('base_model_name', 'unknown'))
        trainer.model.model_spec.config = config_metadata.replace('\\"', "").replace('"', "")
        trainer.model.model_spec.concepts = json.dumps({
            "name": args.concept,
            "images": len(info.samples),
            "buckets": info.buckets,
        })
        del trainer.model.model_spec.hash_sha256
        del trainer.model.model_spec.usage_hint
        trainer.model.model_spec.module = "networks.lora"
        trainer.model.model_spec.tags = json.dumps(tags(args))
        info.metadata = trainer.model.model_spec
        log.debug(f'metadata: {json.dumps(trainer.model.model_spec.__dict__, indent=2)}')
        log.info(f'settings: optimizer={config.optimizer.optimizer} scheduler={config.learning_rate_scheduler} rank={config.lora_rank} alpha={config.lora_alpha} batch={config.batch_size} accumulation={config.gradient_accumulation_steps} epochs={config.epochs}')
        log.info('train: start')
        with pbar if not args.nopbar else contextlib.nullcontext():
            time.sleep(1)
            trainer.train()
        trainer.end()
    except Exception as e:
        log.error(f'train: error={e}')

    if not args.nopbar:
        pbar.remove_task(task)
    del info.metadata
    info.busy = False
    log.debug(f'info: {info}')
    if info.step is None or info.epoch is None:
        info.status = 'failed'
        log.error('train: failed')
        return
    log.info(f'save: {trainer.config.output_model_destination}')
    if info.epoch == args.epochs:
        info.status = 'partial'
        log.info('train: completed')
    else:
        info.status = 'completed'
        log.info('train: completed partial')
