# pylint: disable=no-member

import os
import cv2
import torch
from .util import TrainArgs, Obj
from .config import get_config
from .logger import log


config = None
debug = False
disable = []
_faces = {}


def debug_log(*args, **kwargs):
    if debug:
        log.debug(*args, **kwargs)


def check_dynamicrange(image, epsilon=1e-10):
    tensor = torch.from_numpy(image)
    I_min = torch.min(tensor)
    I_max = torch.max(tensor)
    I_min = torch.clamp(I_min, min=epsilon)
    I_max = torch.clamp(I_max, min=epsilon)
    dynamic_range_db = round(torch.log10(I_max / I_min).item(), 2)
    if dynamic_range_db < config.min_dynamic_range:
        raise ValueError(f'face-range: min={config.min_dynamic_range} res={dynamic_range_db}')


def check_size(image):
    h, w, _c = image.shape
    if h * w < config.min_size:
        debug_log(f'validate image-size={w*h} min-size={config.min_size} image={image.shape}')
        raise ValueError(f'image-size: {w*h}')
    if h < config.min_height:
        debug_log(f'validate image-height={h} min-height={config.min_height}')
        raise ValueError(f'image-height: {h}')
    if w < config.min_width:
        debug_log(f'validate image-width={w} min-width={config.min_width}')
        raise ValueError(f'image-width: {w}')

def check_blur(image):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = 1 / cv2.Laplacian(bw, cv2.CV_64F).var()
    variance = round(variance, 2)
    if variance > config.max_blur:
        debug_log(f'validate face-blur: {variance} max-blur: {config.max_blur}')
        raise ValueError(f'face-blur: {variance}')


def detect_face(image):
    from .face import load, detect
    load()
    all_faces, scores, boxes = detect(image, min_confidence=config.min_face_confidence, max_detected=10, iou=config.max_face_overlap)
    shapes = [face.shape for face in all_faces]
    sizes = [face.shape[0] * face.shape[1] for face in all_faces]
    relative = [round((face.shape[0] * face.shape[1]) / (all_faces[0].shape[0] * all_faces[0].shape[1]), 2) for face in all_faces]
    for i in range(len(all_faces)):
        debug_log(f'validate face-detected: n={i+1}/{len(all_faces)} width={shapes[i][1]} height={shapes[i][0]} score={scores[i]:.2f} box={boxes[i]} size={sizes[i]} relative={relative[i]} image={image.shape}')
    if len(all_faces) == 0:
        debug_log(f'validate face-detected: min-confidence={config.min_face_confidence}')
        all_faces, scores, boxes = detect(image, min_confidence=0, max_detected=3, iou=config.max_face_overlap)
        debug_log(f'validate face-retest: scores={[round(s, 2) for s in scores]} image={image.shape}')
        raise ValueError('face-detected: none')
    if max(scores) < config.min_face_confidence:
        debug_log(f'validate face-confidence: min-confidence={config.min_face_confidence}')
        raise ValueError(f'face-confidence: {max(scores):.2f}')
    large_faces = len([r for r in relative if r > config.max_face_size])
    if large_faces > config.max_faces:
        debug_log(f'validate face-count: all={len(all_faces)} large={large_faces} max-faces={config.max_faces}')
        raise ValueError(f'face-count: {large_faces}')
    face = all_faces[0]
    box = boxes[0]
    h, w, _c = face.shape
    if h < config.min_face_size:
        debug_log(f'validate face-height={h} min-face-height={config.min_face_size}')
        raise ValueError(f'face-height: {h}')
    if w < config.min_face_size:
        debug_log(f'validate face-width={w} min-face-width={config.min_face_size}')
        raise ValueError(f'face-width: {w}')
    face_perc = round((h * w) / (image.shape[0] * image.shape[1]), 2)
    if face_perc < config.min_face_perc and config.min_face_perc > 0:
        debug_log(f'validate face-area={face_perc} min-face-area={config.min_face_perc}')
        raise ValueError(f'face-area: {face_perc}')
    if box[0] == 0 or box[1] == 0 or box[2] == image.shape[1] or box[3] == image.shape[0]:
        debug_log(f'validate face-box: {box}')
        raise ValueError(f'face-box: {box}')
    # if expand[0] == 0 or expand[1] == 0 or expand[2] == image.shape[1] or expand[3] == image.shape[0]:
    #    raise ValueError(f'face-expand: {box}')
    return face


def check_similarity(reference, image):
    if reference is None or config.max_distance <= 0 or 'similarity' in disable:
        return
    from .similarity import distance, init
    init()
    res = distance(reference, image)
    if res < 0:
        disable.append('similarity')
    if res > config.max_distance:
        debug_log(f'validate face-similarity: {round(1 - res, 2)} max-distance={config.max_distance}')
        raise ValueError(f'face-similarity: {round(1 - res, 2)}')


def validate(f, image, args: TrainArgs):
    global config, debug # pylint: disable=global-statement
    config = get_config('validate')
    config = Obj(config)
    try:
        # image = cv2.imread(f)
        if image is None:
            raise ValueError('invalid')
        h, w, _c = image.shape
        if h == 0 or w == 0:
            raise ValueError('empty')
        if not args.validate:
            return None
        if args.debug:
            debug = True
        check_size(image)
        face = detect_face(image)
        _faces[os.path.basename(f)] = face.shape[1], face.shape[0]
        check_blur(face)
        check_dynamicrange(face)
        check_similarity(args.reference, image)
    except Exception as e:
        # from app.logger import console
        # console.print_exception(max_frames=20)
        return { os.path.basename(f): str(e) }
    return face


def faces():
    return _faces
