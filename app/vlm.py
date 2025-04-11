from typing import Union
import os
import re
import time
import json
import torch
import accelerate
import transformers
from PIL import Image
from .logger import log
from .util import login, b64, cache_dir


dtype = torch.bfloat16
processor: transformers.AutoProcessor = None
model: transformers.Gemma3ForConditionalGeneration = None
accelerator = accelerate.Accelerator()
models = {
    '4b': 'google/gemma-3-4b-it',
    '12b': 'google/gemma-3-12b-it',
}
default_question = ''
default_system = """
You are a helpful assistant that can analyze photos.
What are gender, age, hair color, eye color and skin tone that can be used to describe the person in the photo in a simple single sentence?
Always be confident and avoid saying estimate, approximately, appears to be, etc.
Do not add extra adjectives such as stunning striking.
Do not add any description of the clothes or scene.
For every field except "number", only analyze primary person in the image.
Output should be in JSON format with fields:
- age: estimated persons age as a number.
- gender: estimated gender.
- race: estimated race.
- nationality: estimated nationality or heritage.
- eyes: eye color
- skin: skin tone and color.
- hair: hair style, length and color.
- features: describing prominent facial features.
- expression: describing facial expression.
- marks: any distinctive marks such as beauty marks or tattoos.
- nsfw: grade photo as safe, sexy, nude, explict, porn, etc.
- people: number of people with faces clearly visible in the image.
- base: short descriptive sentence about subject that includes gender, age description, hair and eye color and skin color or tone, without features or expression. for example: "woman in early twenties with light blonde hair, blue eyes and fair skin tone".
"""
verbose_system = """
- body: description of the body type and shape.
- makeup: description of the make-up.
- jewlery: description of jewelry, if any.
- resemblance: what is the most similar celebrity or a model and how likely is that similarity.
- obstruction: any items that are directly obstructing the face such as glasses, hands, hair falling over the face, etc.
- quality: grade the quality of the visible face from 0.0 to 1.0 with 1.0 being highest. pay attention to obstructions, visibility, sharpness, and relative size of the face in the image. for example, if face is small and blurry, the quality should be low.
"""
detailed_system = """
- style: overall style and mood of the image
- background: description of the background of the image
- detailed: very detailed description of the subject
"""
default_options = {
    'max_new_tokens': 512,
    'do_sample': False,
    'num_beams': 3,
    'temperature': 0.1,
    'top_k': 0,
    'top_p': 0.0,
}


def set_options(**kwargs):
    if kwargs:
        default_options.update(kwargs)
    return default_options


def analyze(image: Union[str, Image.Image],
            prompt: str = None,
            variant: str = '4b',
            offload: bool = True,
            quant: bool = True,
            verbose: bool = False,
            detailed: bool = False,
            **kwargs,
           ):

    global processor, model # pylint: disable=global-statement
    if model is None:
        quantization_config = None
        model_id = models.get(variant, None)
        if model_id is None:
            log.error(f'vlm load: variant="{variant}" invalid')
            return {}
        if quant:
            try:
                import bitsandbytes as bnb
                quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type= 'nf4')
                log.info(f'vlm quant: bnb={bnb.__version__}')
            except Exception as e:
                log.eror(f'vlm quant: {e}')
        login()
        try:
            t0 = time.time()
            processor = transformers.AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir())
            model = transformers.Gemma3ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_dir(), torch_dtype=dtype, quantization_config=quantization_config)
            t1 = time.time()
            log.info(f'vlm load: model="{model_id}" cache="{cache_dir()}" device="{accelerator.device}" dtype="{dtype}" offload={offload} time={t1-t0:.2f}')
        except Exception as e:
            log.error(f'vlm load: model="{model_id}" {e}')
            return {}
    if model is None:
        return {}

    if isinstance(image, str):
        if os.path.isfile(image):
            try:
                image = Image.open(image)
            except Exception as e:
                log.error(f'vlm input: image="{image}" {e}')
                return {}
        else:
            log.error(f'vlm input: image="{image}" not a file')
            return {}

    if prompt and os.path.exists(prompt):
        with open(prompt, 'r', encoding='utf8') as f:
            prompt = f.read()
    prompt = prompt or default_system
    if verbose:
        prompt += '\n' + verbose_system
    if detailed:
        prompt += '\n' + detailed_system
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": b64(image)},
                {"type": "text", "text": default_question}
            ]
        }
    ]
    options = default_options.copy()
    options.update(kwargs)

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device=accelerator.device, dtype=dtype)
    input_len = inputs["input_ids"].shape[-1]

    response = None
    with torch.inference_mode():
        t0 = time.time()
        model = model.to(accelerator.device)
        generation = model.generate(
            **inputs,
            **options,
        )
        generation = generation[0][input_len:]
        response = processor.decode(generation, skip_special_tokens=True)
        t1 = time.time()
        log.info(f'vlm generate: image={image} {options} len={input_len} time={t1-t0:.2f}')

    if offload:
        model = model.to(device=torch.device("cpu"))
        with torch.cuda.device(accelerator.device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if response is None:
        log.error('vlm generate: no response')
        return {}
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        response = match.group(0)
    try:
        dct = json.loads(response)
    except Exception as e:
        log.error(f'vlm generate: json="{response}" {e}')
        return {}
    dct['resolution'] = round(image.width*image.height/1024/1024, 2)
    return dct


def unload():
    global processor, model # pylint: disable=global-statement
    if model is not None:
        model = model.to(device=torch.device("cpu"))
        with torch.cuda.device(accelerator.device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    processor = None
    model = None
