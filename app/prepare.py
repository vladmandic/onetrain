import os
import time
import uuid
import numpy as np
from PIL import Image
from app.util import TrainArgs, info
from app.logger import log
from app.validate import validate
from app.config import get_config


def prepare(args: TrainArgs):
    info.status = 'prepare'
    from pi_heif import register_heif_opener
    register_heif_opener()
    folder = os.path.join(args.tmp, args.concept)
    os.makedirs(folder, exist_ok=True)
    files = os.listdir(args.input)
    passed = []
    failed = []
    skipped = []
    captions = []
    convert = []
    log.info(f'images prepare: input="{args.input}" output="{folder}"')
    log.info(f'validate: config={get_config("validate") if args.validate else None}')
    t0 = time.time()
    try:
        for file in files:
            status = None
            f = os.path.join(args.input, file)
            ext = os.path.splitext(f)[1].lower()
            image = None
            if ext == '.txt':
                status = None if not args.caption else { file: 'skip caption' }
            if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.heif', '.heic', '.tif', '.tiff', '.bmp', '.gif']:
                status = { file: f'skip {ext}' }
                log.debug(f'validate failed: {status}')
            else:
                try:
                    image = Image.open(f)
                    if image.mode != 'RGB':
                        converted = { os.path.basename(f): str(image.mode) }
                        log.debug(f'validate convert: {converted}')
                        convert.append(converted)
                        image = image.convert('RGB')
                    image = np.array(image)
                except Exception as e:
                    log.debug(f'images: {f} {e}')
                    status = { file: str(e) }
            if image is not None and status is None:
                status = validate(f, image, args)
            if status is not None:
                if any('skip' in v for v in status.values()):
                    skipped.append(status)
                    continue
                else:
                    log.debug(f'validate failed: {status}')
                    failed.append(status)
                    continue
            # shutil.copy(f, tgt)
            image = Image.fromarray(image)
            tgt = os.path.join(folder, str(uuid.uuid4())) + args.format
            image.save(tgt)
            if f.endswith('.txt'):
                captions.append(os.path.basename(f))
            else:
                passed.append(os.path.basename(f))
    except Exception as e:
        log.error(f'images: {e}')
    args.input = folder
    t1 = time.time()
    info.validate = {
        'passed': passed,
        'failed': failed,
        'captions': captions,
        'skipped': skipped,
        'convert': convert,
    }
    log.info(f'validate: pass={len(passed)} fail={len(failed)} captions={len(captions)} skip={len(skipped)} time={t1-t0:.2f}')