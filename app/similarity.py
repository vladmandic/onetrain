import cv2
from deepface import DeepFace
from deepface.modules import verification
from .logger import log


models = [ 'VGG-Face', 'Facenet512', 'ArcFace', 'DeepFace', 'Dlib', 'Facenet', 'SFace', 'OpenFace', 'DeepID', 'GhostFaceNet']
detectors = ['opencv', 'retinaface', 'yolov8', 'centerface', 'mtcnn', 'dlib', 'ssd', 'mediapipe']
metrics = ['cosine', 'euclidean', 'euclidean_l2']
normalization = ['base', 'raw', 'Facenet', 'Facenet2018', 'VGGFace', 'VGGFace2', 'ArcFace']

_actions = [] # ['emotion', 'age', 'gender', 'race']
_detector = 'yolov8' # best-by-far
_model = 'VGG-Face' # alternative: ArcFace
_normalize = 'VGGFace2' # alternative: ArcFace
_metric = 'cosine'
reference_file = None
reference = None
initialized = False


def face(img, detector: str = None, model: str = None, normalize: str = None, actions: list = None):
    try:
        img = cv2.imread(img) if isinstance(img, str) else img
        args = {
            'detector_backend': detector or detectors[0],
            'model_name': model or models[0],
            'normalization': normalize or normalization[0],
            'enforce_detection': False,
            'align': True,
            'expand_percentage': 0,
            'anti_spoofing': False,
            'max_faces': 1,
        }
        represent = DeepFace.represent(img, **args)
        person = represent[0]
        if len(actions) > 0:
            args = {
                'detector_backend': detector or detectors[0],
                'actions': actions or [],
                'enforce_detection': False,
                'align': True,
                'expand_percentage': 0,
                'anti_spoofing': False,
            }
            analyze = DeepFace.analyze(img, **args)
            person.update(analyze[0])
        return person
    except Exception as e:
        log.error(f'embedding: {e}')
        raise ValueError('face-embedding: none') from e


def distance(source, target):
    global reference_file, reference # pylint: disable=global-statement
    if reference_file != source:
        reference_file = source
        reference_image = cv2.imread(source) if isinstance(source, str) else source
        if reference_image is None:
            log.debug('face-reference: none')
            return -1
        reference = face(reference_image, detector=_detector, model=_model, normalize=_normalize, actions=_actions)
        log.info(f'validate similarity: model="{_model}" detector="{_detector}" normalization="{_normalize}" actions={_actions} metric="{_metric}"')
        log.info(f'validate reference: file="{source}" shape={reference_image.shape} data={len(reference["embedding"])} confidence={reference["face_confidence"]}')
    target_image = cv2.imread(target) if isinstance(target, str) else target
    target = face(target_image, detector=_detector, model=_model, normalize=_normalize, actions=_actions)
    if reference is None or 'embedding' not in reference:
        raise ValueError('face-reference: none')
    if target is None or 'embedding' not in target:
        raise ValueError('face-target: none')
    res = verification.find_distance(reference["embedding"], target["embedding"], distance_metric=_metric)
    return round(res, 2)


def init():
    global initialized # pylint: disable=global-statement
    if not initialized:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[1:], 'GPU')
        logical_devices = tf.config.list_logical_devices('GPU')
        log.info(f'TF: physical={physical_devices} logical={logical_devices}')
        initialized = True


def unload():
    from deepface.modules import modeling
    if hasattr(modeling, 'cached_models'):
        for k in list(modeling.cached_models):
            modeling.cached_models[k] = None
        del modeling.cached_models


"""
if __name__ == '__main__':
    import os
    import time
    folder = '/home/vlado/generative/Input/nicole-borda/'
    reference = 'nicole (3).png'
    people = []
    ref = face(os.path.join(folder, reference), detector=_detector, model=_model, normalize=_normalize, actions=_actions)
    rprint(f'reference: file="{reference}" data={len(ref["embedding"])} confidence={ref["face_confidence"]}')
    errors = 0
    rprint(f'detector={_detector} model={_model} normalization={_normalize} start')
    t0 = time.time()
    for f in os.listdir(folder):
        if f.endswith('.jpg') or f.endswith('.png'):
            res = face(os.path.join(folder, f), detector=_detector, model=_model, normalize=_normalize, actions=_actions)
            if res is None or 'embedding' not in res or res["face_confidence"] < 0.1:
                errors += 1
                continue
            distance = round(verification.find_distance(ref["embedding"], res["embedding"], distance_metric=metrics[0]), 2)
            # rprint(f'  embedding: file="{f}" data={len(emb["embedding"])} confidence={emb["face_confidence"]} distance={distance}')
            people.append({'file': f, 'distance': distance, 'confidence': res["face_confidence"]})
    t1 = time.time()
    people = sorted(people, key=lambda x: x['distance'])
    rprint(f'detector={_detector} model={_model} normalization={_normalize} errors={errors} time={t1-t0:.2f}')
    rprint('top   ', people[:10])
    rprint('bottom', people[-10:])
"""
