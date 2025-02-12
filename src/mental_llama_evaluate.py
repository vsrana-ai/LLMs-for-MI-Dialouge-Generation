import argparse
import os
import shutil

import numpy as np
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.long = np.int_
np.unicode = np.unicode_

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from accelerate import Accelerator
from datasets import Dataset
from huggingface_hub import HfApi
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from transformers import (
    EarlyStoppingCallback,
    pipeline,
    TrainingArguments,
    Trainer
)

# allenai/longformer-base-4096
# google/bigbird-roberta-base
# google/long-t5-tglobal-xl
# llama3.2:3b
parser = argparse.ArgumentParser()
parser.add_argument('model', help="Specify the model name to use (e.g., bert-base-uncased)")
parser.add_argument('--folderpath', type=str, default=".", help="Specify the folder path for the CSV files")
parser.add_argument('--use_accelerator', action='store_true', help="Use the accelerator for training")
args = parser.parse_args()

id2label = {0: "low", 1: "high"}
label2id = {"low": 0, "high": 1}

# Load the CSV files (assuming train, validation, and test splits are already provided)
train_df = pl.read_csv(
    os.path.join(args.folderpath, "merged_train.csv"),
    columns=["transcript_id", "utterance_text", "utterance_id", "mi_quality", "interlocutor"]
)
test_df = pl.read_csv(
    os.path.join(args.folderpath, "merged_test.csv"),
    columns=["transcript_id", "utterance_text", "utterance_id", "mi_quality", "interlocutor"]
)


def aggregate_utterances(df):
    return (
        df
        .with_columns(pl.col('interlocutor').str.to_titlecase())
        .with_columns(
            pl.concat_str([pl.col('interlocutor'), pl.col('utterance_text')], separator=': ').alias('utterance_text')
        )
        .select(pl.all().exclude('interlocutor'))
        .group_by('transcript_id', maintain_order=True)
        .agg(
            pl.col('utterance_text').sort_by('utterance_id').str.concat(delimiter=' ; '),
            pl.col('mi_quality').first())
    )

result_file_suffix = '_original'
model_name = args.model

session_level_data_folder = os.path.join(args.folderpath, 'session-level data')
train_df = pl.read_csv(os.path.join(session_level_data_folder, 'train.csv'))
validation_df = pl.read_csv(os.path.join(session_level_data_folder, 'validation.csv'))
test_df = pl.read_csv(os.path.join(session_level_data_folder, 'test.csv'))

save_path = os.path.join(os.pardir, 'hf_output', f"{model_name.replace(os.sep, '-')}_finetuned{result_file_suffix}")
os.makedirs(save_path, exist_ok=True)

def compute_metrics(predictions_df):
    # Compute metrics for single-label classification
    with open(os.path.join(save_path, "classification_metrics.txt"), "w") as f:
        print("Classification Report:", file=f)
        print(classification_report(predictions_df["mi_quality"], predictions_df["mi_quality_pred"], digits=4), file=f)

        balanced_accuracy_score(predictions_df["mi_quality"], predictions_df["mi_quality_pred"])

        cm = confusion_matrix(predictions_df["mi_quality"], predictions_df["mi_quality_pred"], labels=["low","high"])
        print("Confusion Matrix:", file=f)
        print(cm, file=f)

    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["low","high"], yticklabels=["low","high"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.show()

train_df = train_df.with_columns(
    pl.col('mi_quality').map_elements(label2id.__getitem__, return_dtype=pl.Int64)
)
validation_df = validation_df.with_columns(
    pl.col('mi_quality').map_elements(label2id.__getitem__, return_dtype=pl.Int64)
)
test_df = test_df.with_columns(
    pl.col('mi_quality').map_elements(label2id.__getitem__, return_dtype=pl.Int64)
)

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df.to_pandas())
validation_dataset = Dataset.from_pandas(validation_df.to_pandas())
test_dataset = Dataset.from_pandas(test_df.to_pandas())

if model_name == 'klyang/MentaLLaMA-chat-7B':
    from transformers import LlamaTokenizer, LlamaConfig, LlamaForSequenceClassification
    tokenizer_class = LlamaTokenizer
    config_class = LlamaConfig
    model_class = LlamaForSequenceClassification
elif model_name == 'mental/mental-bert-base-uncased':
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("mental/mental-bert-base-uncased")
elif model_name == 'Tianlin668/MentalT5':
    from transformers import T5Tokenizer, T5ForSequenceClassification
    tokenizer = T5Tokenizer.from_pretrained('Tianlin668/MentalT5')
    model = T5ForSequenceClassification.from_pretrained('Tianlin668/MentalT5')


tokenizer = tokenizer_class.from_pretrained(model_name)

config_kwargs = dict(
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    finetuning_task="text-classification",
    problem_type="single_label_classification",
    pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
)
config = config_class.from_pretrained(model_name, **config_kwargs)

print("Using model config:")
print(config)
print("Tokenizer max length from config:", tokenizer.model_max_length)

typical_model_config_fields = ["n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"]
context_windows = [
    tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else np.inf,
    getattr(config, "max_position_embeddings", np.inf)
]
context_windows.extend([getattr(config, field) for field in typical_model_config_fields if hasattr(config, field)])

max_sequnce_length = min(8192, min(context_windows))
# if model_name == supported_models["reformer"]:
#     config.axial_pos_shape = (config.axial_pos_shape[0], max_sequnce_length // config.axial_pos_shape[0])

model = model_class.from_pretrained(model_name, device_map='auto', config=config)

tokenizer_kwargs = dict(
    padding="max_length",
    truncation=True,
    max_length=max_sequnce_length,
)
print("Max input length retrieved from other config fields", context_windows.pop()) if len(context_windows) else print(f"No Max input variable found for {model}")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["utterance_text"],
        **tokenizer_kwargs
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
validation_dataset = validation_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove columns that are not needed for the model
train_dataset = train_dataset.remove_columns(["utterance_text", "transcript_id"])
validation_dataset = validation_dataset.remove_columns(["utterance_text", "transcript_id"])
test_dataset = test_dataset.remove_columns(["utterance_text", "transcript_id"])

# Rename the label column to "labels" (required by Hugging Face)
train_dataset = train_dataset.rename_column("mi_quality", "label")
validation_dataset = validation_dataset.rename_column("mi_quality", "label")
test_dataset = test_dataset.rename_column("mi_quality", "label")

# Set the format for PyTorch
train_dataset.set_format("torch")
validation_dataset.set_format("torch")
test_dataset.set_format("torch")

if args.use_accelerator:
    accelerator = Accelerator(cpu=True)
    model, train_dataset, validation_dataset, test_dataset = accelerator.prepare(model, train_dataset, validation_dataset, test_dataset)

# # Define the training arguments
# training_args = TrainingArguments(
#     eval_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=args.train_batch_size,
#     per_device_eval_batch_size=args.train_batch_size,
#     num_train_epochs=args.epochs,
#     weight_decay=0.01,
#     logging_dir=os.path.join(save_path, "logs"),
#     logging_steps=10,
#     save_strategy="epoch",
#     save_total_limit=1,  # Limit the number of saved checkpoints
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_accuracy",
#     greater_is_better=True,
# )

# def compute_validation_metrics(p):
#     preds = np.argmax(p.predictions, axis=1)
#     labels = p.label_ids
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

# # Define the trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=validation_dataset,
#     processing_class=tokenizer,
#     compute_metrics=compute_validation_metrics,
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
#     preprocess_logits_for_metrics=preprocess_logits_for_metrics
# )

# if args.task == "finetune":
#     # Train the model
#     trainer.train()

# # Evaluate the model on the test set
# metrics = trainer.evaluate(test_dataset)
# print("Test Metrics:", metrics)

# Example inference with the fine-tuned model
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

# Classify a new therapy session
session_text = """
Therapist: How are you feeling today?
Client: I'm feeling very anxious about my upcoming exams.
Therapist: Let's explore that anxiety. What triggers it?
..."""

result = classifier(session_text)
print("Classification Result:", result)

predicted_labels = []
for test_text in test_df["utterance_text"]:
    single_pred = classifier(test_text, **tokenizer_kwargs)[0]["label"]  # returns a list of dicts
    predicted_labels.append(single_pred)

test_df = test_df.with_columns(
    mi_quality=pl.col('mi_quality').map_elements(id2label.__getitem__, return_dtype=pl.String),
    mi_quality_pred=pl.Series(predicted_labels)
)

test_df.write_csv(os.path.join(save_path, "test_with_predictions.csv"))

compute_metrics(test_df)
