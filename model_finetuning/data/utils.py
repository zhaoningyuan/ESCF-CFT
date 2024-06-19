from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

from ..utils.misc import IGNORE_INDEX
from ..utils.logger import get_logger
from functools import partial

import torch


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from ..hparams.args import DataArguments


logger = get_logger(__name__)

def infer_max_len(source_len: int, target_len: int, max_len: int,) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_source_len = max_len - min(max_target_len, target_len)
    return max_source_len, max_target_len

def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
    text_column: str = "prompt",
    label_column: str = "response",
) -> Dict[str, List[List[int]]]:
    assert text_column in examples, f"Column {text_column} not found in the dataset."

    model_inputs = {"input_ids": [], "attention_mask": [], }
     
    if label_column in examples:
        model_inputs.update({"labels": []})
        for text, label in zip(examples[text_column], examples[label_column]):
            # if label is a list, take the first one
            if isinstance(label, list):
                label = label[0]
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            labels = tokenizer.encode(label, add_special_tokens=False)
            max_source_len, max_target_len = infer_max_len(len(input_ids), len(labels), data_args.cutoff_len)
            source_ids = input_ids[:max_source_len]
            target_ids = labels[:max_target_len]
            if data_args.train_on_prompt:
                source_mask = source_ids
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)
            input_ids = source_ids + target_ids
            labels = source_mask + target_ids
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
    else:
        for text in examples[text_column]:
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            max_source_len = min(len(input_ids), data_args.cutoff_len)
            source_ids = input_ids[:max_source_len]
            input_ids += [tokenizer.eos_token_id]
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
    return model_inputs
    # # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    # batch_size = len(examples[text_column])
    # inputs = examples[text_column]
    # # inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    # targets = [str(x) for x in examples[label_column]]
    # model_inputs = tokenizer(inputs)
    # labels = tokenizer(targets, add_special_tokens=False)  # don't add bos token because we concatenate with inputs
    # # input + label + EOS
    # for i in range(batch_size):
    #     sample_input_ids = model_inputs["input_ids"][i]
    #     label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
    #     # print(i, sample_input_ids, label_input_ids)
    #     model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
    #     labels["input_ids"][i] = [IGNORE_INDEX] * len(sample_input_ids) + label_input_ids
    #     model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # # padding
    # for i in range(batch_size):
    #     sample_input_ids = model_inputs["input_ids"][i]
    #     label_input_ids = labels["input_ids"][i]
    #     model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
    #         data_args.cutoff_len - len(sample_input_ids)
    #     ) + sample_input_ids
    #     model_inputs["attention_mask"][i] = [0] * (data_args.cutoff_len - len(sample_input_ids)) + model_inputs[
    #         "attention_mask"
    #     ][i]
    #     labels["input_ids"][i] = [IGNORE_INDEX] * (data_args.cutoff_len - len(sample_input_ids)) + label_input_ids
    #     model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:data_args.cutoff_len])
    #     model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:data_args.cutoff_len])
    #     labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:data_args.cutoff_len])
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    raise NotImplementedError("Packed dataset is not supported yet.")
    


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", training_args: "Seq2SeqTrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:  # Split the dataset
            val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
            dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
            return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}


def get_preprocess_func(
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    tokenizer: "PreTrainedTokenizer",
) -> Tuple[Callable, Callable]:
    if data_args.packing:
        preprocess_func = partial(
            preprocess_packed_supervised_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    else:
        preprocess_func = partial(
            preprocess_supervised_dataset,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    return preprocess_func