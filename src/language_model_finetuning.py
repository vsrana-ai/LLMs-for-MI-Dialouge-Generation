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
parser = argparse.ArgumentParser()
parser.add_argument('model', help="Specify the model name to use (e.g., bert-base-uncased)")
parser.add_argument('task', choices=["finetune", "inference", "evaluate"], help="Specify the task to perform")
parser.add_argument('--folderpath', type=str, default=".", help="Specify the folder path for the CSV files")
parser.add_argument(
    '--augmented_train_file', type=str, default=None,
    help="Specify the name of the augmented train CSV file (e.g., 'augmented_llama3.1:70b.csv')"
)
parser.add_argument('--epochs', type=int, default=50, help="Specify the number of epochs for training")
parser.add_argument('--validation_size', type=float, default=0.1, help="Specify the validation size")
parser.add_argument('--low_mi_quality_validation_size', type=float, default=0.3, help="Specify the validation size for low MI quality")
parser.add_argument('--train_batch_size', type=int, default=8, help="Specify the training batch size")
parser.add_argument('--use_accelerator', action='store_true', help="Use the accelerator for training")
parser.add_argument('--augmentation_type', type=str, default="LLM", help="Specify the type of augmentation to use")
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

if args.augmented_train_file is not None:
    if args.augmentation_type == "LLM":
        augmented_train_df = pl.read_csv(
            os.path.join(args.folderpath, args.augmented_train_file),
            columns=["transcript_id", "utterance_text", "interlocutor"]
        )
    elif args.augmentation_type == "NLP":
        augmented_train_df = pl.read_csv(os.path.join(args.folderpath, args.augmented_train_file))
        if len(augmented_train_df) > len(train_df):
            base_train_df = pl.concat(
                [train_df] * (len(augmented_train_df) // len(train_df)),
                how="vertical"
            )
        else:
            base_train_df = train_df
        augmented_train_df = base_train_df.select(
            pl.all().exclude('utterance_text'),
            utterance_text=augmented_train_df.to_numpy().flatten()
        )
    else:
        raise NotImplementedError(f"Augmentation type {args.augmentation_type} is not supported")
    transcript_to_max_utt_id = dict(
        train_df
        .select(
            pl.col('transcript_id'),
            pl.col('utterance_id').max().over('transcript_id')
        )
        .unique()
        .to_numpy()
    )
    augmented_train_df = (
        augmented_train_df
        .with_columns(
            max_utterance_id=pl.col('transcript_id').map_elements(transcript_to_max_utt_id.__getitem__, return_dtype=pl.Int32)
        )
        .with_columns(
            utterance_id=pl.arange(pl.col('max_utterance_id').first(), pl.col('max_utterance_id').first() + pl.col('transcript_id').len()).over('transcript_id')
        )
        .select(['transcript_id', 'utterance_id', 'utterance_text', 'interlocutor'])
    )

    transcript_to_mi_quality = dict(
        train_df
        .unique(subset=["transcript_id", "mi_quality"])
        .select(["transcript_id", "mi_quality"])
        .to_numpy()
    )
    augmented_train_df = augmented_train_df.with_columns(
        mi_quality=pl.col('transcript_id').map_elements(transcript_to_mi_quality.__getitem__, return_dtype=pl.String)
    )
    result_file_suffix = f'_augmented_{os.path.splitext(os.path.basename(args.augmented_train_file))[0]}'
else:
    result_file_suffix = '_original'

session_level_data_folder = os.path.join(args.folderpath, 'session-level data')
if not os.path.exists(os.path.join(args.folderpath, 'session-level data')):
    train_df = aggregate_utterances(train_df).sample(fraction=1.0, seed=2025)
    validation_samples = round(args.validation_size * len(train_df))
    low_validation_samples = round(args.low_mi_quality_validation_size * validation_samples)
    high_validation_samples = validation_samples - low_validation_samples
    train_partitions, valid_partitions = [], []
    for partition in train_df.partition_by('mi_quality'):
        partition_size = len(partition)
        valid_size = low_validation_samples if partition['mi_quality'].str.contains('low')[0] else high_validation_samples
        valid_partitions.append(partition[-valid_size:])
        train_partitions.append(partition[:-valid_size])

    train_df = pl.concat(train_partitions, how="vertical")
    validation_df = pl.concat(valid_partitions, how="vertical")
    test_df = aggregate_utterances(test_df)

    os.makedirs(session_level_data_folder, exist_ok=True)
    train_df.write_csv(os.path.join(session_level_data_folder, 'train.csv'))
    validation_df.write_csv(os.path.join(session_level_data_folder, 'validation.csv'))
    test_df.write_csv(os.path.join(session_level_data_folder, 'test.csv'))
else:
    train_df = pl.read_csv(os.path.join(session_level_data_folder, 'train.csv'))
    validation_df = pl.read_csv(os.path.join(session_level_data_folder, 'validation.csv'))
    test_df = pl.read_csv(os.path.join(session_level_data_folder, 'test.csv'))

if args.augmented_train_file is not None:
    augmented_train_df = aggregate_utterances(augmented_train_df)
    train_df = pl.concat([train_df, augmented_train_df], how="vertical")

model_name = args.model
save_path = os.path.join(os.pardir, 'hf_output', f"{model_name.replace(os.sep, '-')}_finetuned{result_file_suffix}")
model_save_file = os.path.join(save_path, "model")
own_pretrained_model_name = os.path.join('jackmedda', os.path.basename(save_path)).replace(':', '_').replace('(', '-').replace(')', '-')

def compute_metrics(predictions_df):
    # Compute metrics for single-label classification
    with open(os.path.join(save_path, "classification_metrics.txt"), "w") as f:
        print("Classification Report:", file=f)
        print(classification_report(predictions_df["mi_quality"], predictions_df["mi_quality_pred"], digits=4), file=f)

        print(f'Balanced accuracy score: {balanced_accuracy_score(predictions_df["mi_quality"], predictions_df["mi_quality_pred"])}', file=f)

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

if args.task == "evaluate":
    test_df_with_predictions = pl.read_csv(os.path.join(save_path, "test_with_predictions.csv"))
    compute_metrics(test_df_with_predictions)
else:
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

    supported_models = {
        "longformer": "allenai/longformer-base-4096",
        "bigbird": "google/bigbird-roberta-base",
        "t5": "google-t5/t5-base",
        "longt5": "google/long-t5-tglobal-base",
        "modernbert": "answerdotai/ModernBERT-base"
        # "reformer": "reformer" # "google/reformer-crime-and-punishment"
    }

    # Load the tokenizer for the selected model
    if model_name == supported_models["longformer"]:
        from transformers import LongformerConfig, LongformerTokenizer, LongformerForSequenceClassification
        config_class = LongformerConfig
        tokenizer_class = LongformerTokenizer
        sequence_classification_class = LongformerForSequenceClassification
    elif model_name == supported_models["bigbird"]:
        from transformers import BigBirdConfig, BigBirdTokenizer, BigBirdForSequenceClassification
        config_class = BigBirdConfig
        tokenizer_class = BigBirdTokenizer
        sequence_classification_class = BigBirdForSequenceClassification
    elif model_name == supported_models["t5"]:
        from transformers import T5Config, T5Tokenizer, T5ForSequenceClassification
        config_class = T5Config
        tokenizer_class = T5Tokenizer
        sequence_classification_class = T5ForSequenceClassification
    elif model_name == supported_models["longt5"]:
        from transformers import LongT5Config, AutoTokenizer
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from classifiers import LongT5ForSequenceClassification
        config_class = LongT5Config
        tokenizer_class = AutoTokenizer
        sequence_classification_class = LongT5ForSequenceClassification
    elif model_name == supported_models["modernbert"]:
        from transformers import ModernBertConfig, AutoTokenizer, ModernBertForSequenceClassification
        config_class = ModernBertConfig
        tokenizer_class = AutoTokenizer
        sequence_classification_class = ModernBertForSequenceClassification
    elif model_name == supported_models["reformer"]:
        from transformers import ReformerConfig, ReformerTokenizer, ReformerForSequenceClassification
        config_class = ReformerConfig
        tokenizer_class = ReformerTokenizer
        sequence_classification_class = ReformerForSequenceClassification
    else:
        raise ValueError(f"Unsupported model name: {model_name} not in {supported_models}")

    def model_exists_in_hub(model_id):
        api = HfApi()
        try:
            api.model_info(model_id)
            return True
        except Exception:
            return False

    if args.task == "inference":
        model_name = model_save_file
        if model_exists_in_hub(own_pretrained_model_name):
            model_name = own_pretrained_model_name

    if '/' in model_name:
        tokenizer = tokenizer_class.from_pretrained(model_name)
    else:
        if model_name == "reformer":
            import tempfile
            import sentencepiece as spm

            input_file = 'path/to/your/dataset.txt'
            vocab_size = 32000  # Adjust the vocabulary size as needed
            model_prefix = 'sentencepiece_encoder_for_reformer'

            combined_df = pl.concat([train_df, validation_df, test_df])
            import pdb; pdb.set_trace()

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_file:
                # Save the content of the combined dataframe to the temporary file
                combined_df.write_csv(temp_file.name)
                temp_file_path = temp_file.name

                spm.SentencePieceTrainer.Train(f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size}')

            tokenizer = tokenizer_class(
                vocab_file=f"{model_prefix}.model",
                additional_special_tokens={'pad_token': '[PAD]'}
            )
        else:
            raise NotImplementedError("This non-pretrained model is not supported")
    # if model_name == supported_models["reformer"]:
    #     if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
    #         tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #         tokenizer.pad_token = '[PAD]'

    config_kwargs = dict(
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        finetuning_task="text-classification",
        problem_type="single_label_classification",
        pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    config = config_class.from_pretrained(model_name, **config_kwargs)
    if model_name == supported_models["longt5"]:
        config.classifier_dropout = 0

    print("Using model config:")
    print(config)
    print("Tokenizer max length from config:", tokenizer.model_max_length)

    typical_model_config_fields = ["n_positions", "seq_len", "seq_length", "n_ctx", "sliding_window"]
    context_windows = [
        tokenizer.model_max_length if hasattr(tokenizer, "model_max_length") else np.inf,
        getattr(config, "max_position_embeddings", np.inf) - (2 if model_name == supported_models["longformer"] else 0),  # Subtract 2 for the special tokens
    ]
    context_windows.extend([getattr(config, field) for field in typical_model_config_fields if hasattr(config, field)])

    max_sequnce_length = min(8192, min(context_windows))
    if model_name == supported_models["longt5"]:
        max_sequnce_length = 2048
    # if model_name == supported_models["reformer"]:
    #     config.axial_pos_shape = (config.axial_pos_shape[0], max_sequnce_length // config.axial_pos_shape[0])

    model_kwargs = dict(device_map='auto') if model_name != supported_models["bigbird"] else dict()
    model = sequence_classification_class.from_pretrained(model_name, config=config, **model_kwargs)

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

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=model_save_file,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(save_path, "logs"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,  # Limit the number of saved checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        hub_model_id=own_pretrained_model_name
    )

    def compute_validation_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    def preprocess_logits_for_metrics(logits, labels):
        if "t5" in model_name:
            logits = logits[0]

        return logits


    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_validation_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    if args.task == "finetune":
        # Train the model
        trainer.train()

    # Evaluate the model on the test set
    metrics = trainer.evaluate(test_dataset)
    print("Test Metrics:", metrics)

    # Save the fine-tuned model and tokenizer
    if args.task == "finetune":
        model.save_pretrained(model_save_file)
        tokenizer.save_pretrained(model_save_file)
        try:
            trainer.push_to_hub(model_name=own_pretrained_model_name)
        except Exception as e:
            print(f"Error pushing model to Hugging Face Hub: {e}")

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

    shutil.rmtree(model_save_file)

    compute_metrics(test_df)
