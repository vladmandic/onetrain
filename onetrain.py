#!/usr/bin/env python

import os
import argparse
import tempfile
from app.util import info, accelerator # pylint: disable=unused-import # noqa: F401
from app.validate import config as validation_config # pylint: disable=unused-import # noqa: F401
from app.defaults import concepts as concepts_config # pylint: disable=unused-import # noqa: F401
from app.defaults import config as train_config # pylint: disable=unused-import # noqa: F401


args = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'onetrain')
    parser.add_argument('--concept', required=True, type=str, help='concept name')
    parser.add_argument('--input', required=True, type=str, help='folder with training dataset')
    parser.add_argument("--model", required=False, type=str, help='stable diffusion base model')
    parser.add_argument("--train", default=False, action='store_true', help='run training')
    parser.add_argument("--validate", default=False, action='store_true', help='run image validation')
    parser.add_argument("--caption", default=False, action='store_true', help='run captioning')
    parser.add_argument("--triton", default=False, action='store_true', help='use triton')
    parser.add_argument("--resume", default=False, action='store_true', help='resume training from last backup')
    parser.add_argument("--te", default=False, action='store_true', help='train text encoder')
    parser.add_argument("--bias", default=False, action='store_true', help='use bias correction')
    parser.add_argument("--config", required=False, type=str, help='use specific onetrainer config file')
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
    parser.add_argument("--nopbar", default=False, action='store_true', help='disable progress bar')
    parser.add_argument("--debug", default=False, action='store_true', help='debug logging')
    parser.add_argument('--tmp', default=os.path.join(tempfile.gettempdir(), 'onetrain'), type=str, help='training temp folder')
    args = parser.parse_args()

    if not os.path.isabs(args.tmp):
        args.tmp = os.path.join(os.path.dirname(__file__), args.tmp)
    os.makedirs(args.tmp, exist_ok=True)
    log_file = args.log or os.path.join(args.tmp, 'onetrain.log')

    from app.logger import log, init
    init(log_file)
    if args.debug:
        log.setLevel('DEBUG')
        log.debug('debug logging enabled')
    log.info('onetrain')
    log.info(f'log: {log_file}')
    log.info(f'args: {args}')
    log.info(f'device: {accelerator.device}')
    if not (os.path.exists(args.input) and os.path.isdir(args.input)):
        log.error(f'input folder not found: {args.input}')
        exit(1)
    if not (os.path.exists(args.onetrainer) and os.path.isdir(args.onetrainer)):
        log.error(f'onetrainer folder not found: {args.input}')
        exit(1)
    if not (os.path.exists(args.model) and os.path.isfile(args.model)):
        log.error(f'model not found: {args.model}')
        exit(1)

    from app.caption import caption
    caption(args)
    from app.train import train
    train(args)
