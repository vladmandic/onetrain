import os
import time
import uuid
import contextlib
import statistics
import cv2
import numpy as np
from PIL import Image
from .util import TrainArgs, info, free
from .logger import log
from .validate import validate, faces
from .config import get_config
from .background import remove


passed = []
failed = []
skipped = []
captions = []
convert = []
generate = {}
pairs = {}
buckets = {}
resized = {}
bucketized = {'ar': [], 'w': [], 'h': []}


def resize_image(f, image: Image.Image, read = False) -> Image.Image:
    size_round = get_config('validate').get('size_round', 1)
    resize_longest = get_config('validate').get('resize_longest', 1)
    if min(image.width, image.height) > resize_longest:
        thumb = image.copy()
        if image.width > image.height:
            thumb.thumbnail((image.width, resize_longest))
        else:
            thumb.thumbnail((resize_longest, image.height))
    ar0 = round(image.width / image.height, 1)
    ar1 = 0
    while ar0 != ar1:
        w, h = size_round * (image.width // size_round), size_round * (image.height // size_round)
        ar1 = round(w / h, 1)
        size_round = size_round // 2
        if not read and ((image.width != w) or (image.height != h)):
            if 2 * size_round not in resized:
                resized[2 * size_round] = []
            resized[2 * size_round].append(os.path.basename(f))
    thumb = image.resize((w, h), Image.Resampling.LANCZOS)
    return thumb


def read_image(f: str) -> np.ndarray:
    try:
        image = Image.open(f)
        if image.mode != 'RGB':
            converted = { os.path.basename(f): str(image.mode) }
            log.debug(f'validate convert: {converted}')
            convert.append(converted)
            image = image.convert('RGB')
        resized_image = resize_image(f, image, read=True)
        image = np.array(image)
        resized_image = np.array(image)
    except Exception as e:
        log.debug(f'read image: file="{f}" {e}')
        return None, None
    return image, resized_image


def save_image(image: np.ndarray, file, args: TrainArgs, same=False):
    folder = os.path.join(args.tmp, args.concept)
    image = Image.fromarray(image)
    max_size = get_config('validate').get('max_size', image.width * image.height)
    while image.width * image.height > max_size:
        image = image.resize((int(image.width/1.1), int(image.height/1.1)), Image.Resampling.LANCZOS)
    image = resize_image(file, image)
    if args.rembg:
        image = remove(image)
    if not same:
        tgt = os.path.join(folder, str(uuid.uuid4())) + args.format
        info.images.append(tgt)
    else:
        tgt = os.path.join(folder, file)
    # log.debug(f'image save: {file} -> {tgt}')
    image.save(tgt)
    pairs[os.path.basename(file)] = tgt
    if image.size not in buckets:
        buckets[image.size] = []
    buckets[image.size].append(os.path.basename(file))
    return os.path.basename(tgt)


def diff(w0, h0, w1, h1):
    hi = max(w0*h0, w1*h1)
    lo = min(w0*h0, w1*h1)
    val = 1 - (lo / hi)
    return round(val, 2)


def optimize_buckets(args: TrainArgs, methods=''):
    def update_image(method, image, file):
        buckets[(w, h)].remove(file)
        buckets[ok[0]].append(file)
        bucketized[method].append(file)
        save_image(image, pairs[file], args, same=True)

    if len(methods) == 0:
        return
    methods = methods.split(',')
    methods = [m.strip() for m in methods]
    folder = os.path.join(args.tmp, args.concept)
    threshold = get_config('validate').get('resize_threshold', 0.0)
    for method in methods:
        ok = [k for k, v in buckets.items() if len(v) >= get_config('validate').get('min_bucket', 1)]
        todo = [v for k, v in buckets.items() if len(v) < get_config('validate').get('min_bucket', 1)]
        for bucket in todo:
            for file in bucket:
                f = os.path.join(folder, pairs[file])
                _original, image = read_image(f)
                h, w, _c = image.shape
                if method == 'aspect' or method == 'ar': # based on aspect ratio
                    ok.sort(key=lambda x: abs(x[0] - w) + abs(x[1] - h)) # pylint: disable=cell-var-from-loop
                    ar0 = round(w / h, 1)
                    ar1 = round(ok[0][0] / ok[0][1], 1)
                    difference = diff(w, h, ok[0][0], ok[0][1])
                    if ar0 == ar1 and (difference <= threshold or threshold == 0):
                        image = cv2.resize(image, (ok[0][0], ok[0][1]), interpolation=cv2.INTER_LANCZOS4)
                        update_image(method, image, file)
                    # else:
                    #     log.debug(f'optimize skip: method={method} file="{file}" diff={difference} threshold={threshold} ar={ar0} target={ar1}')
                if method == 'width' or method == 'w': # based on width
                    ok.sort(key=lambda x: abs(x[0] - w)) # pylint: disable=cell-var-from-loop
                    difference = diff(w, h, ok[0][0], ok[0][1])
                    if h >= ok[0][1] and (difference <= threshold or threshold == 0):
                        image = image[0:ok[0][1], 0:w]
                        update_image(method, image, file)
                    # else:
                    #     log.debug(f'optimize skip: method={method} file="{file}" diff={difference} threshold={threshold} h={h} target={ok[0][1]}')
                if method == 'height' or method == 'h': # based on height
                    ok.sort(key=lambda x: abs(x[1] - h)) # pylint: disable=cell-var-from-loop
                    difference = diff(w, h, ok[0][0], ok[0][1])
                    if w >= ok[0][0] and (difference <= threshold or threshold == 0):
                        image = image[0:h, 0:ok[0][0]]
                        update_image(method, image, file)
                    # else:
                    #     log.debug(f'optimize skip: method={method} file="{file}" diff={difference} threshold={threshold} w={w} target={ok[0][0]}')
    for k, v in bucketized.items():
        log.debug(f'optimize: method={k} resized={v}')


def prepare(args: TrainArgs):
    free()
    #init_data()
    passed.clear()
    failed.clear()
    skipped.clear()
    captions.clear()
    convert.clear()
    generate.clear()
    pairs.clear()
    buckets.clear()
    resized.clear()
    bucketized['ar'].clear()
    bucketized['w'].clear()
    bucketized['h'].clear()
    from .logger import pbar
    info.status = 'prepare'
    info.id = args.id
    info.concept = args.concept
    from pi_heif import register_heif_opener
    register_heif_opener()
    folder = os.path.join(args.tmp, args.concept)
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(args.input)
    cfg = get_config('validate')
    log.info(f'images prepare: input="{args.input}" output="{folder}" resize={cfg.get("size_round", 1)}')
    log.info(f'validate: config={get_config("validate") if args.validate else None}')
    t0 = time.time()
    if not args.nopbar:
        task = pbar.add_task(description="prepare", text="", total=len(files))
    try:
        with pbar if not args.nopbar else contextlib.nullcontext():
            for i, file in enumerate(files):
                status = None
                f = os.path.join(args.input, file)
                ext = os.path.splitext(f)[1].lower()
                original = None
                image = None
                if ext == '.txt':
                    status = None if not args.caption else { file: 'skip caption' }
                if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.heif', '.heic', '.tif', '.tiff', '.bmp', '.gif']:
                    status = { file: f'skip {ext}' }
                    log.debug(f'validate failed: {status}')
                else:
                    try:
                        original, image = read_image(f)
                        log.debug(f'validate read: file="{f}" input={list(original.shape)} output={list(image.shape)}')
                    except Exception as e:
                        log.debug(f'images: {f} {e}')
                        status = { file: str(e) }
                if original is not None and status is None:
                    face = validate(f, original, args)
                    if not isinstance(face, np.ndarray):
                        status = face
                    elif cfg.get('gen_portrait', False):
                        if face.shape[1] >= cfg.get('min_width', 1) and face.shape[0] >= cfg.get('min_height', 1):
                            log.debug(f'validate extract: face={list(face.shape)}')
                            fn = save_image(face, file, args)
                            generate[file] = fn
                if status is not None:
                    if any('skip' in v for v in status.values()):
                        skipped.append(status)
                    else:
                        log.debug(f'validate failed: {status}')
                        failed.append(status)
                else:
                    save_image(image, file, args)
                    if f.endswith('.txt'):
                        captions.append(os.path.basename(f))
                    else:
                        passed.append(os.path.basename(f))
                if not args.nopbar:
                    pbar.update(task, completed=i+1, text=f'{i+1}/{len(files)} images')
    except Exception as e:
        log.error(f'images: {e}')
        # traceback.print_exc()
    if not args.nopbar:
        pbar.remove_task(task)

    # post validation optimizations
    info.status = 'optimize'
    r = [{f'{k}px': len(v)} for k, v in resized.items()]
    log.info(f'optimize resize: {r}')
    if args.validate:
        optimize_buckets(args, cfg.get('bucketize', ''))
    for b in list(buckets):
        if len(buckets[b]) == 0:
            del buckets[b]
    b = [{k: len(v)} for k, v in bucketized.items()]
    log.info(f'optimize bucketize: {b}')
    log.info(f'optimize portraits: generate={list(generate)}')

    # gen stats for info
    if args.validate:
        ideal = []
        for k, v in buckets.items():
            ideal += len(v) * [k[0] / k[1]]
            info.buckets[f'{k[0]}x{k[1]}'] = len(v)
            if len(v) < cfg.get('min_bucket', 1):
                msg = f'validate: min-buckets={cfg.get("min_bucket", None)} size={k} bucket={v}'
                info.warnings.append(msg)
                log.info(msg)
        if len(ideal) > 0:
            log.info(f'aspect ratios: avg={sum(ideal)/len(ideal):.1f} mean={statistics.mean(ideal):.1f} min={min(ideal):.1f} max={max(ideal):.1f}')

        min_face = 0
        max_face = 0
        for _k, v in faces().items():
            if v[0] >= cfg.get('min_face_size', 1) or v[1] >= cfg.get('min_face_size', 1):
                min_face += 1
            if v[0] >= cfg.get('min_width', 1) and v[1] >= cfg.get('min_height', 1):
                max_face += 1
        all_faces = len(faces())
        if all_faces == 0:
            msg = 'validate: no-faces'
            info.warnings.append(msg)
            log.warning(msg)
        elif max_face/all_faces < cfg.get('min_portraits', 0):
            msg = f'validate: min_portraits={cfg.get("min_portraits", 0)} portraits={max_face} generated={len(generate)} total={all_faces} perc={round((len(generate)+max_face)/all_faces, 2)}'
            info.warnings.append(msg)
            log.info(msg)
        else:
            msg = f'validate: min-face={min_face} max-face={max_face} generated={generate} total={all_faces}'
            log.info(msg)
    log.info(f'prepare: total={len(info.images)} buckets={info.buckets}')

    args.input = folder
    t1 = time.time()
    info.validation = {
        'passed': passed,
        'failed': failed,
        'captions': captions,
        'skipped': skipped,
        'generate': generate,
        'convert': convert,
        'pairs': pairs,
        'buckets': buckets,
        'resized': resized,
        'bucketized': bucketized,
    }
    from .face import unload as face_unload
    face_unload()
    from .similarity import unload as similarity_unload
    similarity_unload()
    from .background import unload as background_unload
    background_unload()
    free()
    info.status = "validated"
    log.info(f'validate: pass={len(passed)} fail={len(failed)} generate={len(generate)} captions={len(captions)} skip={len(skipped)} time={t1-t0:.2f}')
