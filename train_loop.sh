#!/usr/bin/env bash

# List of config files (adjust these to your actual 5 configs)
configs=(
  "vlad-lora.json"
  "cassie-lora.json"
  "flux-lora-stable.json"
  "rank2-lora.json"
  "config.json"
)

# List of input datasets (adjust these to your actual 3 datasets)
datasets=(
  "geo-carmilla-pruned"
  "carmilla-pruned"
  "carmilla-flux-prune"
)

# Where your onetrainer script is located
onetrainer_path="/home/hello@moodmagic.ai/onetrainer"

# Base model
model_name="black-forest-labs/FLUX.1-dev"

# Loop over each config/dataset pair
for config in "${configs[@]}"; do
    config_basename="$(basename "$config" .json)"  # e.g. "vlad-lora" from "vlad-lora.json"

    for dataset in "${datasets[@]}"; do
        dataset_basename="$(basename "$dataset")"

        # Construct unique filenames/paths
        log_file="/mnt/sdnext-shared/train-dir/${dataset_basename}_${config_basename}.log"
        output_file="/mnt/sdnext-shared/data-dir/models/Lora/${dataset_basename}_${config_basename}.safetensors"
        tmp_dir="/mnt/sdnext-shared/train-dir/tmp_${dataset_basename}_${config_basename}"
        concept="${dataset_basename}_${config_basename}--train"

        # Run training under nohup so it keeps going if you disconnect
        nohup python onetrain.py \
          --onetrainer "$onetrainer_path" \
          --model "$model_name" \
          --input "/mnt/sdnext-shared/concept-dir/training_datasets/${dataset}" \
          --log "$log_file" \
          --output "$output_file" \
          --tmp "$tmp_dir" \
          --concept "$concept" \
          --config "$config" \
          --type flux \
          --sample \
          --debug \
          --caption \
          > "/mnt/sdnext-shared/train-dir/nohup_${dataset_basename}_${config_basename}.out" 2>&1 &
        
        # Optional: If you want to run each training job sequentially (one after another),
        # remove the "&" above and uncomment "wait" below.
        #
        # wait
    done
done
