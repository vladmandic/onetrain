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
    parser.add_argument('--id', required=False, type=str, help='id')
    parser.add_argument('--concept', required=False, type=str, help='concept name')
    parser.add_argument('--input', required=False, type=str, help='folder with training dataset')
    parser.add_argument("--model", required=False, type=str, help='stable diffusion base model')
    parser.add_argument('--format', default='.jpg', type=str, help='image format')
    parser.add_argument('--reference', required=False, type=str, help='reference image for similarity checks')
    parser.add_argument("--noclean", default=False, action='store_true', help='clean output concept folder on start')
    parser.add_argument("--train", default=False, action='store_true', help='run training')
    parser.add_argument("--tag", default=False, action='store_true', help='add tagging info')
    parser.add_argument("--validate", default=False, action='store_true', help='run image validation')
    parser.add_argument("--caption", default=False, action='store_true', help='run captioning')
    parser.add_argument("--triton", default=False, action='store_true', help='use triton')
    parser.add_argument("--resume", default=False, action='store_true', help='resume training from last backup')
    parser.add_argument("--te", default=False, action='store_true', help='train text encoder')
    parser.add_argument("--bias", default=False, action='store_true', help='use bias correction')
    parser.add_argument("--config", required=False, type=str, help='use specific config file')
    parser.add_argument("--type", required=False, choices=['sd', 'sdxl', 'flux'], help='model type')
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
    parser.add_argument("--sample", default=False, action='store_true', help='enable sampling during training')
    parser.add_argument("--rembg", default=False, action='store_true', help='background removal')
    parser.add_argument("--nopbar", default=False, action='store_true', help='disable progress bar')
    parser.add_argument("--debug", default=False, action='store_true', help='debug logging')
    parser.add_argument('--tmp', default=os.path.join(tempfile.gettempdir(), 'onetrain'), type=str, help='training temp folder')
    args, _unknown = parser.parse_known_args()

    if not os.path.isabs(args.tmp):
        args.tmp = os.path.join(os.path.dirname(__file__), args.tmp)
    os.makedirs(args.tmp, exist_ok=True)
    log_file = args.log or os.path.join(args.tmp, 'onetrain.log')
    init_logger(log_file)
    # os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold:0.25,max_split_size_mb:512')
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
    if not (os.path.exists(args.model) and os.path.isfile(args.model)):
        log.error(f'model not found: {args.model}')
        exit(1)
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
