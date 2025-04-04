#!/usr/bin/env bash

ONETRAIN_PATH=/home/vlado/dev/onetrain
ONETRAINER_PATH=~/branches/onetrainer/
TRAIN_CONFIG=/home/vlado/dev/onetrain/examples/lora-sdxl.json
TMP_PATH=/home/vlado/dev/onetrain/tmp
MODEL_FILE=/mnt/models/stable-diffusion/base/sdxl-base-v10-vaefix.safetensors
INPUT_BASE_PATH=~/generative/Input
OUTPUT_PATH=/mnt/models/Lora
HF_HUB_CACHE=/mnt/models/huggingface

# inputs are in INPUT_BASE_PATH/ACTOR output is in OUTPUT_PATH/ACTOR.safetensors 
ACTOR="$1"
REFERENCE_IMAGE="$2"

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

cd "${ONETRAINER_PATH}"
"${ONETRAIN_PATH}"/onetrain.py \
  --onetrainer "$ONETRAINER_PATH" \
  --model "${MODEL_FILE}" \
  --config "${TRAIN_CONFIG}" \
  --concept "${ACTOR}" \
  --reference "${INPUT_BASE_PATH}/${ACTOR/$REFERENCE_IMAGE}" \
  --input "${INPUT_BASE_PATH}/${ACTOR}/" \
  --output "${OUTPUT_PATH}/${ACTOR}.safetensors" \
  --tmp "${TMP_PATH}" \
  --log "${TMP_PATH}/onetrain.log" \
  --type sdxl \
  --sample \
  --interval 10 \
  --debug \
