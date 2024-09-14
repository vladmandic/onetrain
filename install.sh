#!/usr/bin/env bash
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
uv pip install --requirement requirements.txt
uv pip install --upgrade --no-deps xformers torchvision
pip install -e git+https://github.com/Nerogar/mgds.git#egg=mgds
