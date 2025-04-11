#!/usr/bin/env bash

# NOTE: for flux, MODEL_FILE is REPO_ID on Huggingface, not local file
# Inputs are in INPUT_BASE_PATH/ACTOR output is in OUTPUT_PATH/ACTOR.safetensors 

ONETRAIN_PATH=/home/vlado/dev/onetrain
ONETRAINER_PATH=~/branches/onetrainer/
TRAIN_CONFIG=/home/vlado/dev/onetrain/examples/lora-flux.json
TMP_PATH=/home/vlado/dev/onetrain/tmp
MODEL_FILE=black-forest-labs/FLUX.1-dev
INPUT_BASE_PATH=~/generative/Input
OUTPUT_PATH=/mnt/models/Lora
HF_HUB_CACHE=/mnt/models/huggingface

TRIGGER="woman"
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
  --trigger "${TRIGGER}" \
  --reference "${INPUT_BASE_PATH}/${ACTOR/$REFERENCE_IMAGE}" \
  --input "${INPUT_BASE_PATH}/${ACTOR}/" \
  --output "${OUTPUT_PATH}/${ACTOR}.safetensors" \
  --tmp "${TMP_PATH}" \
  --log "${TMP_PATH}/onetrain.log" \
  --type flux \
  --sample \
  --save \
  --interval 10 \
  --debug \
