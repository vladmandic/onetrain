#!/usr/bin/env bash

ONETRAINER=~/branches/onetrainer/
INPUT=~/generative/Input
MODEL=/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors
OUTPUT=/mnt/models/Lora
ACTOR="$@"

if [ -z "${ACTOR}" ]; then
    echo "Error: actor is not set"
    exit 1
fi

if [[ -f venv/bin/activate ]]
then
    source venv/bin/activate
else
    echo "Error: Cannot activate python venv"
    exit 1
fi

source venv/bin/activate
CUR="${PWD}"
cd "${ONETRAINER}"
"${CUR}"/onetrain.py \
  --onetrainer $ONETRAINER \
  --model $MODEL \
  --config my.json \
  --concept $ACTOR \
  --input $INPUT/$ACTOR/ \
  --output $OUTPUT/$ACTOR.safetensors \
