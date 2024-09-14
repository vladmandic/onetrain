# pylint: disable=no-member

import os
import cv2
import torch
from app.util import TrainArgs, Obj
from app.defaults import validate as validate_config


config = Obj(validate_config)


def dynamicrange(image, epsilon=1e-10):
    tensor = torch.from_numpy(image)
    I_min = torch.min(tensor)
    I_max = torch.max(tensor)
    I_min = torch.clamp(I_min, min=epsilon)
    I_max = torch.clamp(I_max, min=epsilon)
    dynamic_range_db = 20 * torch.log10(I_max / I_min)
    if dynamic_range_db < config.min_dynamic_range:
        raise ValueError(f'range: {dynamic_range_db}')


def size(img):
    h, w, _c = img.shape
    if h < config.min_height or w < config.min_width or h * w < config.min_size:
        raise ValueError(f'size: {w} {h}')


def blur(image):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = 1 / cv2.Laplacian(bw, cv2.CV_64F).var()
    if variance > config.max_blur:
        raise ValueError(f'blur: {variance}')


def detect(image):
    import app.face
    app.face.load()
    faces = app.face.detect(image, min_confidence=config.min_face_confidence, max_detected=config.max_faces)
    if len(faces) == 0:
        raise ValueError('face: none')
    face = faces[0]
    h, w, _c = face.shape
    if h < config.min_face_size or w < config.min_face_size:
        raise ValueError(f'face: {w} {h}')
    return face


def validate(f: str, args: TrainArgs):
    try:
        img = cv2.imread(f)
        if img is None:
            raise ValueError('invalid')
        h, w, _c = img.shape
        if h == 0 or w == 0:
            raise ValueError('empty')
        if not args.validate:
            return None
        size(img)
        face = detect(img)
        blur(face)
        dynamicrange(face)
    except Exception as e:
        # from app.logger import console
        # console.print_exception(max_frames=20)
        return { os.path.basename(f): str(e) }
    return None
