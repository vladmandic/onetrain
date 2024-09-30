#!/usr/bin/env bash
export UV_INDEX_STRATEGY=unsafe-any-match
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install --requirement requirements.txt
uv pip install --upgrade --no-deps xformers
pip install -e git+https://github.com/Nerogar/mgds.git#egg=mgds
