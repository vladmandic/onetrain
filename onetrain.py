#!/usr/bin/env python
import os
import sys
import argparse
import tempfile

# hack to allow cli usage of onetrain
__package__ = os.path.basename(os.path.dirname(__file__)) # pylint: disable=redefined-builtin
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)

from .app.util import info, accelerator # pylint: disable=unused-import
from .app.config import get_config, init_config # pylint: disable=unused-import # noqa: F401
from .app.logger import log, init_logger
from .app.prepare import prepare
from .app.caption import caption
from .app.train import train
from .app.util import TrainArgs, free


args = TrainArgs()


def main():
    info.status = 'init'
    global args # pylint: disable=global-statement
    parser = argparse.ArgumentParser(description = 'onetrain')
    parser.add_argument('--concept', required=False, type=str, help='concept name')
    parser.add_argument('--type', required=False, choices=['sd', 'sdxl', 'flux'], help='model type')
    parser.add_argument('--author', required=False, type=str, help='train author to be included in metadata')
    parser.add_argument('--trigger', required=False, type=str, help='list of triggers to be included in metadata')

    group_path = parser.add_argument_group('Paths')
    group_path.add_argument('--model', required=False, type=str, help='stable diffusion base model')
    group_path.add_argument('--input', required=False, type=str, help='folder with training dataset')
    group_path.add_argument('--output', required=False, type=str, help='specify output location')
    group_path.add_argument('--onetrainer', required=False, type=str, help='path to onetrainer')
    group_path.add_argument('--log', required=False, type=str, help='specify log file')
    group_path.add_argument('--tmp', default=os.path.join(tempfile.gettempdir(), 'onetrain'), type=str, help='training temp folder')

    group_path = parser.add_argument_group('Optional')
    parser.add_argument('--config', required=False, type=str, help='use specific config file')
    group_path.add_argument('--reference', required=False, type=str, help='reference image for similarity checks')

    group_ops = parser.add_argument_group('Operations')
    group_ops.add_argument('--validate', default=False, action='store_true', help='run image validation')
    group_ops.add_argument('--caption', default=False, action='store_true', help='run captioning')
    group_ops.add_argument('--tag', default=False, action='store_true', help='add tagging info')
    group_ops.add_argument('--train', default=False, action='store_true', help='run training')
    group_ops.add_argument('--sample', default=False, action='store_true', help='enable sampling during training')
    group_ops.add_argument('--rembg', default=False, action='store_true', help='background removal')
    group_ops.add_argument('--save', required=False, action='store_true', help='save intermittent steps')
    group_ops.add_argument('--backup', required=False, action='store_true', help='create training backups')
    group_ops.add_argument('--resume', default=False, action='store_true', help='resume training from last backup')
    group_ops.add_argument('--interval', required=False, type=int, help='interval for sample/backup/save')

    group_params = parser.add_argument_group('Parameters')
    group_params.add_argument('--te', default=False, action='store_true', help='train text encoder')
    group_params.add_argument('--nobias', default=False, action='store_true', help='do not use bias correction')
    group_params.add_argument('--epochs', required=False, type=int, help='number of training epochs')
    group_params.add_argument('--accumulation', required=False, type=int, help='gradient accumulation steps')
    group_params.add_argument('--optimizer', required=False, type=str, help='training optimizer')
    group_params.add_argument('--scheduler', required=False, type=str, help='training scheduler')
    group_params.add_argument('--lr', required=False, type=float, help='learning rate')
    group_params.add_argument('--d', required=False, type=float, help='d coefficient')
    group_params.add_argument('--dropout', required=False, type=float, help='dropout probability')
    group_params.add_argument('--warmup', required=False, type=int, help='warmup steps')
    group_params.add_argument('--rank', required=False, type=int, help='lora rank')
    group_params.add_argument('--alpha', required=False, type=int, help='lora alpha')
    group_params.add_argument('--batch', required=False, type=int, help='training batch size')
    group_params.add_argument('--resolution', required=False, type=int, help='training override resolution')

    group_internal = parser.add_argument_group('Internal')
    group_internal.add_argument('--id', required=False, type=str, help='id')
    group_internal.add_argument('--format', default='.jpg', type=str, help='image format')
    group_internal.add_argument('--debug', default=False, action='store_true', help='debug logging')
    group_internal.add_argument('--noclean', default=False, action='store_true', help='clean output concept folder on start')
    group_internal.add_argument('--triton', default=False, action='store_true', help='use triton')
    group_internal.add_argument('--nogradient', default=False, action='store_true', help='disable gradient checkpointing')
    group_internal.add_argument('--nopbar', default=False, action='store_true', help='disable progress bar')
    args, _unknown = parser.parse_known_args()

    if not os.path.isabs(args.tmp):
        args.tmp = os.path.join(os.path.dirname(__file__), args.tmp)
    os.makedirs(args.tmp, exist_ok=True)
    log_file = args.log or os.path.join(args.tmp, 'onetrain.log')
    init_logger(log_file)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.50,max_split_size_mb:512')
    log.info('onetrain')
    log.info(f'log: {log_file}')
    init_config(args)
    log.info(f'args: {args}')
    log.info(f'device: {accelerator.device}')
    free()


main()


if __name__ == '__main__':
    if not (os.path.exists(args.input) and os.path.isdir(args.input)):
        log.error(f'input folder not found: {args.input}')
        exit(1)
    if not (os.path.exists(args.onetrainer) and os.path.isdir(args.onetrainer)):
        log.error(f'onetrainer folder not found: {args.input}')
        exit(1)
    # if not (os.path.exists(args.model) and os.path.isfile(args.model)):
    #     log.error(f'model not found: {args.model}')
    #     exit(1)
    if os.path.exists(os.path.join(args.tmp, args.concept)):
        if args.noclean:
            log.warning(f'concept folder exists: {os.path.join(args.tmp, args.concept)}')
        else:
            log.warning(f'cleaning concept folder: {os.path.join(args.tmp, args.concept)}')
            removed = []
            for f in os.listdir(os.path.join(args.tmp, args.concept)):
                fn = os.path.join(args.tmp, args.concept, f)
                if os.path.isfile(fn) or os.path.islink(fn):
                    removed.append(fn)
                    os.remove(os.path.join(args.tmp, args.concept, f))
            log.debug(f'cleaning concept folder: removed={removed}')
    if args.concept is None:
        log.error('concept name not provided')
        exit(1)
    if args.input is None:
        log.error('input folder not provided')
        exit(1)
    if args.debug:
        log.setLevel('DEBUG')
        log.debug('debug logging enabled')
    try:
        prepare(args)
        caption(args)
        train(args)
    except KeyboardInterrupt:
        log.error('interrupted')
