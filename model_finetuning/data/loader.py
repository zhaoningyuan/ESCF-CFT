import os
import sys
from typing import TYPE_CHECKING, Literal, Optional, Union, Dict

from datasets import load_dataset

from ..utils.logger import get_logger
from ..data.utils import get_preprocess_func

if TYPE_CHECKING:
    from ..hparams import DataArguments

logger = get_logger(__name__)

def get_dataset(
    training_args: "Seq2SeqTrainingArguments",
    data_args: "DataArguments",
    tokenizer: "PreTrainedTokenizer",
):
    dataset = load_dataset(
        "json",
        data_files=data_args.dataset_path,
        split=data_args.split,
    )
    # Shuffle and sample the dataset
    if data_args.max_samples :
        shuffled_dataset = dataset.shuffle(seed=42)
        dataset = shuffled_dataset.select(range(data_args.max_samples))  
    with training_args.main_process_first(desc="pre-process dataset"):
        column_names = list(next(iter(dataset)).keys())
        preprocess_func = get_preprocess_func(data_args, training_args, tokenizer)
        kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                desc="Running tokenizer on dataset",
            )
        dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
    return dataset


