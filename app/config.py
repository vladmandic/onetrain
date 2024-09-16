import os
import json
from app.logger import log


config = {}


def get_config(section: str = None):
    if section is None:
        return config
    if section not in config:
        log.error(f'config: section={section} not found')
        return {}
    return config.get(section, {})


def init_config(args):
    main_config = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(main_config, 'r', encoding='utf8') as f:
        global config # pylint: disable=global-statement
        try:
            config = json.load(f)
        except Exception as e:
            log.error(f'config: file="{main_config}" {e}')
            return
    log.info(f'config: main="{main_config}" sections={list(config)}')
    user_config = args.config
    if user_config is not None:
        if not os.path.isabs(user_config):
            user_config = os.path.join(os.path.dirname(__file__), '..', user_config)
        if not os.path.exists(user_config):
            log.error(f'config file {user_config} not found')
        else:
            with open(user_config, 'r', encoding='utf8') as f:
                dct = {}
                try:
                    dct = json.load(f)
                except Exception as e:
                    log.error(f'config: file="{user_config}" {e}')
                log.info(f'config: user="{user_config}" sections={list(dct)}')
                # config = merge(config, dct)
                if 'train' in dct:
                    for k, v in dct['train'].items():
                        if k in args:
                            setattr(args, k, v)
                    del dct['train']
                config.update(dct)
