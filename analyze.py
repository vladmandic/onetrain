#!/usr/bin/env python
import os
import sys
import time
import json
from PIL import Image


# hack to allow cli usage of analyze
__package__ = os.path.basename(os.path.dirname(__file__)) # pylint: disable=redefined-builtin
parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent)


from .app.logger import log, init_logger
log.info('vlm')


def vlm(images: list,
        verbose:bool=False,
        detailed:bool=False,
        variant:str='4b',
        prompt:str=None,
        do_sample:bool=None,
        temperature:float=None,
        top_k:int=None,
        top_p:float=None,
        num_beams:int=None,
        offload:bool=False,
        quant:bool=False,
       ) -> dict:
    from .app.vlm import set_options, get_options
    from .app.vlm import analyze as run_analyze
    init_logger()
    if images is None or len(images) == 0:
        log.error('vlm input: no image specified')
        return { 'error': 'no image specified' }
    if num_beams:
        set_options(num_beams=num_beams)
    if temperature:
        set_options(temperature=temperature)
    if do_sample:
        set_options(do_sample=do_sample)
    if top_k:
        set_options(top_k=top_k)
    if top_p:
        set_options(top_p=top_p)
    if isinstance(images, str) or isinstance(images, Image.Image):
        images = [images]
    results = []
    if verbose:
        log.info(f'vlm options: {get_options()}')
    for image in images:
        dct = run_analyze(image, offload=offload, variant=variant, quant=quant, verbose=verbose, detailed=detailed, prompt=prompt)
        results.append(dct)
    return results


if __name__ == "__main__":
    import argparse
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
        exit(1)

    from .app.vlm import unload
    t0 = time.time()
    for fn in args.images:
        log.info(f'vlm input: image="{fn}"')
        res = vlm(images=fn,
                  verbose=args.verbose,
                  detailed=args.detailed,
                  do_sample=args.sample,
                  temperature=args.temperature,
                  variant=args.variant,
                  top_k=args.top_k,
                  top_p=args.top_p,
                  num_beams=args.beams,
                  offload=args.offload,
                  quant=args.quant,
                  prompt=args.prompt,
            )
        if res:
            rprint(json.dumps(res, indent=4))
    t1 = time.time()
    log.info(f'vlm time: {t1-t0:.2f}')
    unload()
