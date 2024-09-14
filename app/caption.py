import os
import shutil
import contextlib
from app.util import TrainArgs, set_path, info, accelerator
from app.logger import log, pbar
from app.validate import validate, config


def copy(args: TrainArgs):
    folder = os.path.join(args.tmp, args.concept)
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(args.input)
    passed = []
    failed = []
    skipped = []
    captions = []
    log.info(f'images copy: input="{args.input}" output="{folder}"')
    log.info(f'validate: config={config.__dict__ if args.validate else None}')
    try:
        for file in files:
            f = os.path.join(args.input, file)
            if f.endswith('.txt'):
                status = None if not args.caption else { file: 'skip caption' }
            else:
                status = validate(f, args)
            if status is not None:
                if any('skip' in v for v in status.values()):
                    skipped.append(status)
                    continue
                else:
                    failed.append(status)
                    continue
            tgt = os.path.join(folder, file)
            shutil.copy(f, tgt)
            if f.endswith('.txt'):
                captions.append(f)
            else:
                passed.append(f)
    except Exception as e:
        log.error(f'images: {e}')
    args.input = folder
    log.info(f'validate: pass={len(passed)} fail={len(failed)} captions={len(captions)} skip={len(skipped)}')
    log.debug(f'validate failed: {failed}')


def caption_onetrainer(args: TrainArgs, tagger: str = ''):
    set_path(args)
    import torch

    def caption_progress_callback(current, total):
        if not args.nopbar:
            pbar.update(task, completed=current, total=total, text=f'{current}/{total} images')


    if tagger == 'blip':
        log.info(f'caption: model="Salesforce/blip2-opt-2.7b" path={args.input}')
        from modules.module.Blip2Model import Blip2Model # pylint: disable=import-error
        model = Blip2Model(device=accelerator.device, dtype=torch.float16)
        info.busy = True
        if not args.nopbar:
            task = pbar.add_task(description="caption blip", text="", total=0)
        with pbar if not args.nopbar else contextlib.nullcontext():
            model.caption_folder(
                sample_dir=args.input,
                initial_caption='',
                caption_prefix='',
                caption_postfix=', ',
                mode='add',
                error_callback=lambda fn: log.error(f'caption: path={fn}'),
                include_subdirectories=False,
                progress_callback=caption_progress_callback,
            )
        if not args.nopbar:
            pbar.remove_task(task)

    if tagger == 'wd14':
        log.info(f'caption: model="SmilingWolf/wd-v1-4-vit-tagger-v2" path={args.input}')
        from modules.module.WDModel import WDModel # pylint: disable=import-error
        model = WDModel(device=accelerator.device, dtype=torch.float16)
        if not args.nopbar:
            task = pbar.add_task(description="caption wd14", text="", total=0)
        with pbar if not args.nopbar else contextlib.nullcontext():
            model.caption_folder(
                sample_dir=args.input,
                initial_caption='',
                caption_prefix='',
                caption_postfix=', ',
                mode='add',
                error_callback=lambda fn: log.error(f'caption: path={fn}'),
                include_subdirectories=False,
                progress_callback=caption_progress_callback,
            )
        info.busy = False
        if not args.nopbar:
            pbar.remove_task(task)

    model = None


def caption_wdtagger(args: TrainArgs):
    from transformers import pipeline
    folder = os.path.join(args.tmp, args.concept)
    model = "p1atdev/wd-swinv2-tagger-v3-hf"
    log.info(f'caption: model="{model}" path="{folder}"')
    pipe = pipeline(
        "image-classification",
        model=model,
        trust_remote_code=True,
        device=accelerator.device,
    )
    files = os.listdir(folder)
    files = [f for f in files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    if not args.nopbar:
        task = pbar.add_task(description="caption wdtagger", text="", total=len(files))
    with pbar if not args.nopbar else contextlib.nullcontext():
        for i, f in enumerate(files):
            file = os.path.join(folder, f)
            items = pipe(file, top_k=15)
            tags = []
            for item in items:
                k, v = item['label'], item['score']
                if 'rating:sensitive' in k or v > 0.05:
                    k = k.replace(' ', '_').replace('rating:', '')
                    tags.append(k)
            # log.debug(f'caption: "{f}"={tags}')
            tag = os.path.splitext(file)[0] + '.txt'
            with open(tag, 'a', encoding='utf8') as f:
                txt = ', '.join(tags)
                f.write(f'{txt}, ')
            if not args.nopbar:
                pbar.update(task, completed=i+1, text=f'{i+1}/{len(files)} images')
    if not args.nopbar:
        pbar.remove_task(task)

    model = None


def caption_promptgen(args):
    import cv2
    import transformers

    folder = os.path.join(args.tmp, args.concept)
    repo = "MiaoshouAI/Florence-2-base-PromptGen-v1.5"
    log.info(f'caption: model="{repo}" path="{folder}"')

    model = transformers.AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
    model = model.to(accelerator.device)
    processor = transformers.AutoProcessor.from_pretrained("MiaoshouAI/Florence-2-base-PromptGen-v1.5", trust_remote_code=True)
    prompt = "<MORE_DETAILED_CAPTION>"

    files = os.listdir(folder)
    files = [f for f in files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    if not args.nopbar:
        task = pbar.add_task(description="caption promptgen", text="", total=len(files))
    with pbar if not args.nopbar else contextlib.nullcontext():
        for i, f in enumerate(files):
            file = os.path.join(folder, f)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(accelerator.device)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                do_sample=False,
                num_beams=3
            )
            generated = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(generated, task=prompt, image_size=(image.shape[0], image.shape[1]))
            prompt = parsed.get(prompt, '')
            prompt = prompt.split('\n')[0].replace('\\(', '').replace('\\)', '').strip()
            # log.debug(f'caption: "{f}"={prompt}')
            tag = os.path.splitext(file)[0] + '.txt'
            with open(tag, 'a', encoding='utf8') as f:
                f.write(prompt)
            if not args.nopbar:
                pbar.update(task, completed=i+1, text=f'{i+1}/{len(files)} images')
    if not args.nopbar:
        pbar.remove_task(task)

    model = None
    processor = None


def caption(args: TrainArgs):
    copy(args)
    if not args.caption:
        return
    from app.defaults import caption as captioners
    folder = os.path.join(args.tmp, args.concept)
    log.info(f'caption: config={captioners}')

    for captioner in captioners:
        log.info(f'caption: run={captioner} folder="{folder}"')
        if captioner == 'clear':
            for f in os.listdir(folder):
                if f.endswith('.txt'):
                    os.remove(os.path.join(folder, f))
        if captioner == 'concept':
            log.info(f'caption: concept="{args.concept}"')
            for f in os.listdir(folder):
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    fn = os.path.splitext(f)[0] + '.txt'
                    with open(os.path.join(folder, fn), 'a', encoding='utf8') as file:
                        file.write(f'{args.concept}, ')
        if captioner == 'wdtagger':
            caption_wdtagger(args)
        if captioner == 'promptgen':
            caption_promptgen(args)
        if captioner == 'blip':
            caption_onetrainer(args, 'blip')
        if captioner == 'wd14':
            caption_onetrainer(args, 'wd14')

    for f in os.listdir(folder): # fix formatting
        if f.endswith('.txt'):
            fn = os.path.join(folder, f)
            with open(fn, 'r', encoding='utf8') as file:
                tag = file.read()
            tag = tag.replace('\n', ' ').replace('  ', ' ').replace(' ,', ',')[:-2].strip()
            with open(fn, 'w', encoding='utf8') as file:
                file.write(tag)
