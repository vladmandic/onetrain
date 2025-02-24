#!/usr/bin/env bash
export UV_INDEX_STRATEGY=unsafe-any-match
python -m venv venv
source venv/bin/activate
pip install uv
uv pip install --requirement requirements.txt
uv pip install --upgrade --no-deps xformers
uv pip install flash_attn --no-build-isolation
uv pip uninstall onnxruntime
uv pip uninstall xformers
uv pip uninstall mgds
pip install -e git+https://github.com/Nerogar/mgds.git@ae2b07f38451e60f72b6f9b9dba1ddd79463c437#egg=mgds
echo "OneTrainer must be installed manually and set to supported version: eb8fe41"
# echo "example: git clone https://github.com/Nerogar/OneTrainer/ /tmp/onetrainer; cd /tmp/onetrainer; git checkout eb8fe41"
