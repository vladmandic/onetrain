import os
from .logger import log
from .util import accelerator


model = None


def load():
    global model # pylint: disable=global-statement
    if model is None:
        name = 'yolov8n-face.pt'
        log.info(f'validate face: model="{name}"')
        from ultralytics import YOLO # pylint: disable=import-outside-toplevel
        model_url = os.path.join(os.path.dirname(__file__), '..', 'models', name)
        model = YOLO(model_url)
        model = model.to(accelerator.device)


def unload():
    global model # pylint: disable=global-statement
    if model is not None:
        model = None


def detect(image, imgsz: int = 640, half: bool = True, augment: bool = True, agnostic: bool = False, retina: bool = False, min_confidence: float = 0.6, iou: float = 0.5, max_detected: int = 10):
    predictions = model.predict(
        source=[image],
        stream=False,
        verbose=False,
        imgsz=imgsz,
        half=half,
        device=accelerator.device,
        augment=augment,
        agnostic_nms=agnostic,
        retina_masks=retina,
        conf=min_confidence,
        iou=iou,
        max_det=max_detected,
    )
    result = []
    for prediction in predictions:
        boxes = prediction.boxes.xyxy.detach().int().cpu().numpy() if prediction.boxes is not None else []
        scores = prediction.boxes.conf.detach().float().cpu().numpy() if prediction.boxes is not None else []
        scores = [round(score, 2) for score in scores]
        for _score, box in zip(scores, boxes):
            box = box.tolist()
            expand = (box[2] - box[0]) // 4, (box[3] - box[1]) // 4
            box = [max(0, box[0] - expand[0]), max(0, box[1] - expand[1]), min(image.shape[1], box[2] + expand[0]), min(image.shape[0], box[3] + expand[1])]
            face = image[box[1]:box[3], box[0]:box[2]]
            result.append(face)
    return result, scores
