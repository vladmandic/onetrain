from .logger import log

model = "u2net" # "silueta", "u2net", "u2net_human_seg", "isnet-general-use",
providers = ['CUDAExecutionProvider']
args = {
    'only_mask': False,
    'post_process_mask': False,
    'bgcolor': None,
    'alpha_matting': False,
    'alpha_matting_foreground_threshold': 240, # 0..255
    'alpha_matting_background_threshold': 10, # 0..255
    'alpha_matting_erode_size': 10, # 0..40
    'session': None,
}


def remove(image):
    import onnxruntime as ort
    import rembg
    args['data'] = None
    if args['session'] is None:
        args['session'] = rembg.new_session(model, providers)
        log.info(f'remove background: model={model} providers={ort.get_available_providers()}') # pylint: disable=c-extension-no-member
        log.info(f'remove background: {args}')
    args['data'] = image
    output = rembg.remove(**args)
    output = output.convert('RGB')
    return output


def unload():
    args['session'] = None
