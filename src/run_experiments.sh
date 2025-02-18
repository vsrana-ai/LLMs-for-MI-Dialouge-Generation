#!/usr/bin/env bash
# filepath: run_experiments.sh
# Usage example: ./run_experiments.sh 50 4

# Pass epochs and train_batch_size as arguments
EPOCHS=$1
TASK=$2

# Paths
FOLDER_PATH="/home/gmedda/projects/MI-LLMAugmentation/AnnoMI Data/Experimental Setup 1 (single transcript in train)/unprocessed"
LLM_AUGMENTED_PATH="/home/gmedda/projects/MI-LLMAugmentation/AnnoMI Data/LLM Augmented Dataset/chatgpt_prompt_engineering_prompt_v1/Experimental Setup 1 (single transcript in train)/unprocessed"
NLP_AUGMENTED_PATH="/home/gmedda/projects/MI-LLMAugmentation/AnnoMI Data/NLP Augmented Dataset/Experimental Setup 1 (single transcript in train)/unprocessed"
OUTPUT_PATH="/home/gmedda/projects/MI-LLMAugmentation/hf_output"

# List of supported models
MODELS=(
  "allenai/longformer-base-4096"
  "google/bigbird-roberta-base"
  # "google-t5/t5-base"
  "answerdotai/ModernBERT-base"
  # "google/long-t5-tglobal-base"
)

# For each model, run once without augmented data, then with each augmented file
for MODEL in "${MODELS[@]}"
do
  case "$MODEL" in
    "allenai/longformer-base-4096")
      TRAIN_BATCH_SIZE=4
      ;;
    "google/bigbird-roberta-base")
      TRAIN_BATCH_SIZE=4
      ;;
    "google-t5/t5-base")
      TRAIN_BATCH_SIZE=16
      ;;
    "answerdotai/ModernBERT-base")
      TRAIN_BATCH_SIZE=4
      ;;
    "google/long-t5-tglobal-base")
      TRAIN_BATCH_SIZE=2
      ;;
    *)
      TRAIN_BATCH_SIZE=2
      ;;
  esac

  OUTPUT_ORIGINAL_PATH="${OUTPUT_PATH}/${MODEL//\//-}_finetuned_original"
  if [[ ! -d "$OUTPUT_ORIGINAL_PATH" || "$TASK" == "evaluate" ]]; then
    echo "Running finetuning on $MODEL without augmented data..."
    CUDA_VISIBLE_DEVICES=1 python language_model_finetuning.py "$MODEL" $TASK \
      --folderpath "$FOLDER_PATH" \
      --epochs "$EPOCHS" \
      --train_batch_size "$TRAIN_BATCH_SIZE" \
      --use_accelerator
  fi

  echo "Running finetuning on $MODEL with all LLM augmented data..."
  while IFS= read -r -d '' AUG_FILE
  do
    AUG_FILE_BASENAME=$(basename "$AUG_FILE" .csv)
    OUTPUT_FINETUNED_PATH="${OUTPUT_PATH}/${MODEL//\//-}_finetuned_augmented_${AUG_FILE_BASENAME}"
    if [[ -d "$OUTPUT_FINETUNED_PATH" && "$TASK" == "finetune" ]]; then
      echo "Output path $OUTPUT_FINETUNED_PATH already exists. Skipping..."
      continue
    fi

    echo "Using augmented file: $AUG_FILE"
    CUDA_VISIBLE_DEVICES=1 python language_model_finetuning.py "$MODEL" $TASK \
      --folderpath "$FOLDER_PATH" \
      --augmented_train_file "$AUG_FILE" \
      --epochs "$EPOCHS" \
      --train_batch_size "$TRAIN_BATCH_SIZE" \
      --augmentation_type "LLM" \
      --use_accelerator
  done < <(find "$LLM_AUGMENTED_PATH" -name 'augmented_*.csv' -print0)
done

for MODEL in "${MODELS[@]}"
do
  case "$MODEL" in
    "allenai/longformer-base-4096")
      TRAIN_BATCH_SIZE=4
      ;;
    "google/bigbird-roberta-base")
      TRAIN_BATCH_SIZE=4
      ;;
    "google-t5/t5-base")
      TRAIN_BATCH_SIZE=16
      ;;
    "meta-llama/Llama-3.2-3B")
      TRAIN_BATCH_SIZE=8
      ;;
    "reformer")
      TRAIN_BATCH_SIZE=16
      ;;
    *)
      TRAIN_BATCH_SIZE=2
      ;;
  esac

  echo "Running finetuning on $MODEL with all NLP augmented data..."
  while IFS= read -r -d '' AUG_FILE
  do
    if [[ ! "$AUG_FILE" =~ \([23457]\) ]]; then
      echo "Skipping $AUG_FILE as it does not contain a required number in the pattern (2), (3), (4), (5), or (7)."
      continue
    fi

    AUG_FILE_BASENAME=$(basename "$AUG_FILE" .csv)
    OUTPUT_FINETUNED_PATH="${OUTPUT_PATH}/${MODEL//\//-}_finetuned_augmented_${AUG_FILE_BASENAME}"
    if [[ -d "$OUTPUT_FINETUNED_PATH" && "$TASK" == "finetune" ]]; then
      echo "Output path $OUTPUT_FINETUNED_PATH already exists. Skipping..."
      continue
    fi

    echo "Using augmented file: $AUG_FILE"
    CUDA_VISIBLE_DEVICES=1 python language_model_finetuning.py "$MODEL" $TASK \
      --folderpath "$FOLDER_PATH" \
      --augmented_train_file "$AUG_FILE" \
      --epochs "$EPOCHS" \
      --train_batch_size "$TRAIN_BATCH_SIZE" \
      --augmentation_type "NLP" \
      --use_accelerator
  done < <(find "$NLP_AUGMENTED_PATH" -name 'augmented_*.csv' -print0)
done