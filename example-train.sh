#!/usr/bin/env bash

ONETRAINER=~/branches/onetrainer/
INPUT=~/generative/Input
MODEL=/mnt/models/stable-diffusion/sdxl/TempestV0.1-Artistic.safetensors
OUTPUT=/mnt/models/Lora
CONFIG=example-config.json
TYPE=sdxl
ACTOR="$1"
REFERENCE="$2"

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

CUR="${PWD}"
echo "ACTOR: $ACTOR"
echo "REFERENCE: $REFERENCE"
echo "MODEL: $MODEL"
echo "INPUT: $INPUT"
echo "OUTPUT: $OUTPUT"
echo "CONFIG: $CUR/$CONFIG"

cd "${ONETRAINER}"
"${CUR}"/onetrain.py \
  --onetrainer "$ONETRAINER" \
  --model "$MODEL" \
  --type $TYPE \
  --config "$CUR/$CONFIG" \
  --concept "$ACTOR" \
  --reference "$INPUT/$ACTOR/$REFERENCE" \
  --input "$INPUT/$ACTOR/" \
  --output "$OUTPUT/$ACTOR.safetensors" \
  --sample \
