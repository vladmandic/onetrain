import os
import time
import contextlib
import cv2
import torch
from .util import TrainArgs, set_path, info, accelerator, free
from .logger import log
from .config import get_config


all_tags = []
first_prompt = ''
dtype = torch.float16


def caption_onetrainer(args: TrainArgs, tagger: str = ''):
    set_path(args)
    from .logger import pbar
    task = None

    def caption_progress_callback(current, total):
        if not args.nopbar and task is not None:
            pbar.update(task, completed=current, total=total, text=f'{current}/{total} images')


    if tagger == 'blip':
        log.info(f'caption: model="Salesforce/blip2-opt-2.7b" path={args.input}')
        from modules.module.Blip2Model import Blip2Model # pylint: disable=import-error
        model = Blip2Model(device=accelerator.device, dtype=dtype)
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
        model = WDModel(device=accelerator.device, dtype=dtype)
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
    import transformers
    from .logger import pbar
    threshold = get_config('tag')
    folder = os.path.join(args.tmp, args.concept)
    repo = "p1atdev/wd-swinv2-tagger-v3-hf"
    log.info(f'caption: model="{repo}" path="{folder}" threshold={threshold}')
    model = transformers.AutoModelForImageClassification.from_pretrained(repo, trust_remote_code=True)
    processor = transformers.AutoImageProcessor.from_pretrained(repo, trust_remote_code=True, use_fast=False)
    model = model.to(device=accelerator.device, dtype=dtype) # wdtagger somehow runs faster on cpu
    files = os.listdir(folder)
    files = [f for f in files if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    if not args.nopbar:
        task = pbar.add_task(description="caption wdtagger", text="", total=len(files))
    with pbar if not args.nopbar else contextlib.nullcontext():
        for i, f in enumerate(files):
            file = os.path.join(folder, f)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor.preprocess(image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs.to(model.device, model.dtype))
            logits = torch.sigmoid(outputs.logits[0]) # take the first logits
            results = { model.config.id2label[i]: logit.float() for i, logit in enumerate(logits) }
            results = { k: v.item() for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True) if v > threshold }
            words = list(results)
            tag = os.path.splitext(file)[0] + '.txt'
            with open(tag, 'a', encoding='utf8') as f:
                txt = ', '.join(words).replace('rating:', '')
                f.write(f'{txt}, ')
                log.debug(f'caption: "{tag}"="{txt}"')
            if not args.nopbar:
                pbar.update(task, completed=i+1, text=f'{i+1}/{len(files)} images')
    if not args.nopbar:
        pbar.remove_task(task)

    model = None
    processor = None


def caption_florence(args, repo, task_prompt: str = "<MORE_DETAILED_CAPTION>"):
    import transformers
    from .logger import pbar

    folder = os.path.join(args.tmp, args.concept)
    log.info(f'caption: model="{repo}" path="{folder}"')

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        processor = transformers.AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        model = model.to(device=accelerator.device, dtype=dtype)
    except Exception as e:
        log.error(f'caption: model="{repo}" error={e}')
        return

    files = os.listdir(folder)
    files = [f for f in files if os.path.splitext(f)[1].lower() == args.format]
    if not args.nopbar:
        task = pbar.add_task(description="caption florence", text="", total=len(files))
    with pbar if not args.nopbar else contextlib.nullcontext():
        for i, f in enumerate(files):
            file = os.path.join(folder, f)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device=model.device, dtype=model.dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=True,
                num_beams=3
            )
            generated = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(generated, task=task_prompt, image_size=(image.shape[0], image.shape[1]))
            generated_ids, generated, image = None, None, None
            p = parsed.get(task_prompt, '')
            p = p.split('\n')[0].replace('\\(', '').replace('\\)', '').replace(' a ', ' ').replace('A ', '').replace('The ', '').replace('  ', ' ').strip()
            tag = os.path.splitext(file)[0] + '.txt'
            with open(tag, 'a', encoding='utf8') as f:
                f.write(p + '  ')
                log.debug(f'caption: type="cog" "{tag}"="{p}"')
            if not args.nopbar:
                pbar.update(task, completed=i+1, text=f'{i+1}/{len(files)} images')
    if not args.nopbar:
        pbar.remove_task(task)

    model = None
    processor = None


def caption(args: TrainArgs):
    global first_prompt # pylint: disable=global-statement
    info.status = 'caption'
    free()
    all_tags.clear()
    captioners = get_config('caption') if args.caption else []
    folder = os.path.join(args.tmp, args.concept)
    os.makedirs(args.tmp, exist_ok=True)
    log.info(f'caption: config={captioners}')

    t0 = time.time()
    for captioner in captioners:
        # log.info(f'caption: run={captioner} folder="{folder}"')
        if captioner == 'clear':
            for f in os.listdir(folder):
                if f.endswith('.txt'):
                    os.remove(os.path.join(folder, f))
        if captioner == 'trigger':
            for f in os.listdir(folder):
                ext = os.path.splitext(f)[1].lower()
                if ext in [args.format]:
                    fn = os.path.splitext(f)[0] + '.txt'
                    with open(os.path.join(folder, fn), 'a', encoding='utf8') as file:
                        file.write(f'{args.trigger}, ')
        if captioner == 'concept':
            for f in os.listdir(folder):
                ext = os.path.splitext(f)[1].lower()
                if ext in [args.format]:
                    fn = os.path.splitext(f)[0] + '.txt'
                    with open(os.path.join(folder, fn), 'a', encoding='utf8') as file:
                        file.write(f'{args.concept}, ')
        if captioner == 'wdtagger':
            caption_wdtagger(args)
        if captioner == 'promptgen':
            caption_florence(args, "MiaoshouAI/Florence-2-base-PromptGen-v2.0", '<MORE_DETAILED_CAPTION>')
        if captioner == 'promptgen-tags':
            caption_florence(args, "MiaoshouAI/Florence-2-base-PromptGen-v2.0", '<GENERATE_TAGS>')
        if captioner == 'promptgen-aio':
            caption_florence(args, "MiaoshouAI/Florence-2-base-PromptGen-v2.0", '<MIXED_CAPTION>')
        if captioner == 'florence':
            caption_florence(args, "microsoft/Florence-2-base")
        if captioner == 'cog':
            caption_florence(args, "thwri/CogFlorence-2.2-Large")
        if captioner == 'blip':
            caption_onetrainer(args, 'blip')
        if captioner == 'wd14':
            caption_onetrainer(args, 'wd14')
    t1 = time.time()
    if len(captioners) > 0:
        log.info(f'caption: time={t1-t0:.2f}')

    for f in os.listdir(folder): # fix formatting
        if f.endswith('.txt'):
            fn = os.path.join(folder, f)
            with open(fn, 'r', encoding='utf8') as file:
                tag = file.read()
                if first_prompt == '':
                    first_prompt = tag
            tag = tag.replace('\n', ' ').replace('  ', ' ').replace(' ,', ',')[:-2].strip()
            all_tags.append(tag)
            with open(fn, 'w', encoding='utf8') as file:
                file.write(tag)


def tags(args: TrainArgs):
    info.status = 'tag'
    _triggers = {}
    _tags = {}
    _concept = { args.concept: -1 }
    if args.trigger and isinstance(args.trigger, str):
        _triggers = {t.strip(): -1 for t in args.trigger.split(',')}
        log.info(f'caption: triggers={_triggers}')
    if args.tag and isinstance(args.tag, float):
        t0 = time.time()
        count = len(all_tags)
        threshold = get_config('tag') * count
        all_text = ', '.join(all_tags)
        all_words = [w.strip() for w in all_text.split(',')]
        all_tags.clear()
        all_tags.extend([w for w in all_words if ' ' not in w and len(w) > 1])
        _tags = {item: all_tags.count(item) for item in set(all_tags)}
        _tags = {k: v for k, v in sorted(_tags.items(), key=lambda item: item[1], reverse=True) if v >= threshold }
        _tags.pop(args.concept, None)
        t1 = time.time()
        log.info(f'caption: theshold={threshold:.2f} text={len(all_text)} words={len(all_words)} tags={len(all_tags)} time={t1-t0:.2f}')
        log.info(f'caption: tags={_tags}')
        return _tags
    res = { **_triggers, **_concept, **_tags }
    return res


def prompt():
    return first_prompt
