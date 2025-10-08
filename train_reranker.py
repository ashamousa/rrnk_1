"""Training script for document reranking using Hugging Face Transformers.

This script is intentionally self contained so that it can be copy/pasted into
new projects that rely on JSONL datasets for retrieval augmented generation
(RAG) document reranking tasks.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    import evaluate
except ImportError as exc:  # pragma: no cover - informative error for missing dep.
    raise ImportError(
        "The `evaluate` package is required for metric computation. Install it via"
        " `pip install evaluate`."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Holds model related configuration."""

    model_name_or_path: str = field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        metadata={"help": "Model identifier from huggingface.co/models."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to cache pre-trained models."}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use a fast tokenizer."}
    )


@dataclass
class DataConfig:
    """Holds dataset related configuration."""

    train_file: str = field(metadata={"help": "Path to the training jsonl file."})
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "Optional path to validation jsonl."}
    )
    query_column: str = field(
        default="query", metadata={"help": "Column name containing the query."}
    )
    document_column: str = field(
        default="document",
        metadata={"help": "Column name containing the document text."},
    )
    label_column: str = field(
        default="label", metadata={"help": "Column name containing labels."}
    )
    title_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional column for document titles, concatenated before text.",
        },
    )
    max_length: int = field(
        default=512, metadata={"help": "Maximum tokenized sequence length."}
    )
    template: str = field(
        default="Query: {query}\nDocument: {document}",
        metadata={"help": "Template for combining query and document."},
    )


@dataclass
class RerankerArguments:
    """Aggregates all argument groups for the script."""

    model: ModelConfig
    data: DataConfig
    training: TrainingArguments


def parse_arguments() -> RerankerArguments:
    parser = HfArgumentParser((ModelConfig, DataConfig, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return RerankerArguments(model=model_args, data=data_args, training=training_args)


def load_jsonl_dataset(data_args: DataConfig) -> DatasetDict:
    """Load JSONL datasets into a :class:`DatasetDict`.

    Each JSONL entry should contain the columns specified in ``data_args``.
    """

    data_files: Dict[str, str] = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file

    extension = os.path.splitext(data_args.train_file)[1]
    if extension not in {".json", ".jsonl"}:
        raise ValueError(
            "Only `.json` or `.jsonl` files are supported. Received:"
            f" {data_args.train_file}"
        )

    raw_datasets = load_dataset("json", data_files=data_files)
    logger.info("Loaded dataset splits: %s", raw_datasets)
    return raw_datasets


def build_examples(example: Dict[str, str], data_args: DataConfig) -> Dict[str, str]:
    """Combine query and document fields using the provided template."""

    query = example[data_args.query_column]
    document = example[data_args.document_column]
    if data_args.title_column and data_args.title_column in example:
        document = f"{example[data_args.title_column]}\n{document}"

    combined = data_args.template.format(query=query, document=document)
    example["combined_text"] = combined
    return example


def main() -> None:
    args = parse_arguments()
    model_args, data_args, training_args = args.model, args.data, args.training

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in {0, -1} else logging.WARN,
    )

    if training_args.seed is not None:
        set_seed(training_args.seed)

    raw_datasets = load_jsonl_dataset(data_args)
    tokenized_datasets = raw_datasets.map(
        build_examples,
        fn_kwargs={"data_args": data_args},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    def tokenize_function(example: Dict[str, str]) -> Dict[str, list[int]]:
        return tokenizer(
            example["combined_text"],
            truncation=True,
            max_length=data_args.max_length,
        )

    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

    label_column = data_args.label_column
    if label_column not in tokenized_datasets["train"].column_names:
        raise ValueError(
            f"Label column '{label_column}' not found in the dataset."
        )

    label_list = sorted(set(tokenized_datasets["train"][label_column]))
    num_labels = len(label_list)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.squeeze(predictions)
        if predictions.ndim > 1 and predictions.shape[-1] > 1:
            predicted_labels = np.argmax(predictions, axis=-1)
        else:
            probs = 1 / (1 + np.exp(-predictions))
            predicted_labels = (probs > 0.5).astype(int)
        metrics = metric_accuracy.compute(
            predictions=predicted_labels, references=labels
        )
        f1_metrics = metric_f1.compute(
            predictions=predicted_labels, references=labels, average="binary"
        )
        metrics.update({"f1": f1_metrics["f1"]})
        return metrics

    train_dataset = tokenized_datasets["train"].with_format("torch")
    eval_dataset = None
    if "validation" in tokenized_datasets:
        eval_dataset = tokenized_datasets["validation"].with_format("torch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()

    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and "test" in tokenized_datasets:
        predictions = trainer.predict(tokenized_datasets["test"].with_format("torch"))
        if trainer.is_world_process_zero():
            output_path = os.path.join(training_args.output_dir, "predictions.jsonl")
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for pred, label in zip(predictions.predictions, predictions.label_ids):
                    f.write(
                        json.dumps({"logits": pred.tolist(), "label": int(label)})
                        + "\n"
                    )


if __name__ == "__main__":
    main()
