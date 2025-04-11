#!/usr/bin/env python
import os
import sys
import json
import argparse

# hack to allow cli usage of onetrain
__package__ = os.path.basename(os.path.dirname(__file__)) # pylint: disable=redefined-builtin
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)

from .app.logger import log, init_logger


if __name__ == "__main__":
    log.info('vlm')
    init_logger()
    from rich import print as rprint
    from pi_heif import register_heif_opener
    register_heif_opener()

    parser = argparse.ArgumentParser(description = 'onetrain')
    parser.add_argument('--quant', required=False, action='store_true', help='quantize model')
    parser.add_argument('--offload', required=False, action='store_true', help='offload model')
    parser.add_argument('--verbose', required=False, action='store_true', help='verbose output')
    parser.add_argument('--detailed', required=False, action='store_true', help='detailed output')
    parser.add_argument('--variant', required=False, type=str, default='4b', choices=['4b', '12b'], help='model variant')
    parser.add_argument('--prompt', required=False, type=str, default=None, help='prompt or path to prompt file')
    parser.add_argument('--temperature', required=False, type=float, default=None, help='llm temperature')
    parser.add_argument('--beams', required=False, type=int, default=None, help='llm number of beams')
    parser.add_argument('--sample', required=False, action='store_true', help='llm do sample')
    parser.add_argument('--top_k', required=False, type=int, default=None, help='llm top-k')
    parser.add_argument('--top_p', required=False, type=float, default=None, help='llm top-p')
    parser.add_argument('images', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if len(args.images) == 0:
        log.error('vlm input: no image specified')

    from .app.vlm import set_options, analyze, unload
    if args.beams:
        set_options(num_beams=args.beams)
    if args.temperature:
        set_options(temperature=args.temperature)
    if args.sample:
        set_options(do_sample=args.sample)
    if args.top_k:
        set_options(top_k=args.top_k)
    if args.top_p:
        set_options(top_p=args.top_p)
    for fn in args.images:
        log.info(f'vlm input: image="{fn}"')
        res = analyze(fn, offload=args.offload, variant=args.variant, quant=args.quant, verbose=args.verbose, detailed=args.detailed, prompt=args.prompt)
        if res:
            rprint(json.dumps(res, indent=4))

    unload()
