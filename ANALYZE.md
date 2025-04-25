**vlm-analyze** should be ready for use

## Prompt

System prompt is massive and includes a lot of instructions  
For futher tuning, refer to `vlm.py`: `default_system`, `verbose_system`, `detailed_system`

## CLI

using shell wrapper:  
> `./analyze.sh --help`  
 
or activate `venv` and run:  
> `python analyze.py --help`  

## API
```py
import analyze
lst = analyze.vlm(images=..., verbose=True, detailed=True, ...)
```
- in **onetrain** repo  
- new top-level `analyze.py` file with single function `vlm`  
- `images` param is required, all others are optional  
- `images` param can be *list* or *single entry*  
- each entry can be *str:path-to-image*, *str:base64-encoded* or *image:PIL.Image*  
- i suggest running with `verbose=True` and `detailed=True`  
- the rest of params are fine-tuning params and can be left as default  
- key field is `base`, thats the field thats intended as prompt suffix  
- everything else is value-add if you want to use/store/display  
  in the future things like `marks` could be useful as it would note presence and location of things like tattoos, scars, etc.
- returns list-of-dicts (one dict per image)  
  *note*: fields vary depending if `verbose` and `detailed` are set  

example:
```json
[
    {
        "age": 25,
        "gender": "female",
        "race": "Caucasian",
        "nationality": "Unknown",
        "eyes": "brown",
        "skin": "fair",
        "hair": "long, flowing, light blonde",
        "features": "prominent cheekbones, delicate nose",
        "expression": "serene",
        "marks": "none",
        "nsfw": "safe",
        "people": 1,
        "base": "Woman in her mid twenties with long light blonde hair, brown eyes and fair skin.",
        "body": "slim",
        "makeup": "subtle eye makeup, choker",
        "jewelry": "necklace",
        "resemblance": "Similar to IU, 70%",
        "obstruction": "none",
        "quality": 0.85,
        "style": "portrait, artistic",
        "background": "dark, blurred",
        "detailed": "The subject is a young woman with a serene expression. She has long, flowing light blonde hair and brown eyes. Her skin is fair, and she is wearing a black bra and lingerie set with a delicate choker and necklace. The image has a soft, artistic quality.",
        "resolution": 0.25
    }
]
```
