# pylint: disable=no-member

import os
import cv2
import torch
from .util import TrainArgs, Obj
from .config import get_config


config = None
disable = []
_faces = {}


def check_dynamicrange(image, epsilon=1e-10):
    tensor = torch.from_numpy(image)
    I_min = torch.min(tensor)
    I_max = torch.max(tensor)
    I_min = torch.clamp(I_min, min=epsilon)
    I_max = torch.clamp(I_max, min=epsilon)
    dynamic_range_db = round(torch.log10(I_max / I_min).item(), 2)
    if dynamic_range_db < config.min_dynamic_range:
        raise ValueError(f'face-range: {dynamic_range_db}')


def check_size(img):
    h, w, _c = img.shape
    if h * w < config.min_size:
        raise ValueError(f'image-size: {w*h}')
    if h < config.min_height:
        raise ValueError(f'image-height: {h}')
    if w < config.min_width:
        raise ValueError(f'image-width: {w}')

def check_blur(image):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = 1 / cv2.Laplacian(bw, cv2.CV_64F).var()
    variance = round(variance, 2)
    if variance > config.max_blur:
        raise ValueError(f'face-blur: {variance}')


def detect_face(image):
    from .face import load, detect
    load()
    all_faces, scores, boxes = detect(image, min_confidence=config.min_face_confidence / 2, max_detected=10)
    if len(all_faces) == 0:
        raise ValueError('face-detected: none')
    if max(scores) < config.min_face_confidence:
        raise ValueError(f'face-confidence: {max(scores):.2f}')
    if len(all_faces) > config.max_faces:
        raise ValueError(f'face-count: {len(all_faces)}')
    face = all_faces[0]
    box = boxes[0]
    h, w, _c = face.shape
    if h < config.min_face_size:
        raise ValueError(f'face-height: {h}')
    if w < config.min_face_size:
        raise ValueError(f'face-width: {w}')
    face_perc = round((h * w) / (image.shape[0] * image.shape[1]), 2)
    if face_perc < config.min_face_perc and config.min_face_perc > 0:
        raise ValueError(f'face-area: {face_perc}')
    if box[0] == 0 or box[1] == 0 or box[2] == image.shape[1] or box[3] == image.shape[0]:
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
        raise ValueError(f'face-similarity: {round(1 - res, 2)}')


def validate(f, image, args: TrainArgs):
    global config # pylint: disable=global-statement
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
