import re
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, LoraModel, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..utils.logger import get_logger

from ..utils.misc import find_all_linear_modules 


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ..hparams.args import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def _setup_full_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    cast_trainable_params_to_fp32: bool,
) -> None:
    logger.info("Fine-tuning method: Full")

    for name, param in model.named_parameters():
        if cast_trainable_params_to_fp32:
            param.data = param.data.to(torch.float32)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    cast_trainable_params_to_fp32: bool,
) -> None:
    logger.info("Fine-tuning method: Freeze")
    config = model.config

    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "num_layers", None)
        or getattr(config, "n_layer", None)
    )
    if not num_layers:
        raise ValueError("Current model does not support freeze tuning.")

    if finetuning_args.freeze_trainable_layers > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers)
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])

    trainable_layers = []
    for module_name in finetuning_args.freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                "Module {} is not found, please choose from {}".format(module_name, ", ".join(hidden_modules))
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(".{:d}.{}".format(idx, module_name if module_name != "all" else ""))

    if finetuning_args.freeze_extra_modules:
        for module_name in finetuning_args.freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    "Module {} is not found, please choose from {}".format(module_name, ", ".join(non_hidden_modules))
                )

            trainable_layers.append(module_name)


    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers) :
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info("Set trainable layers: {}".format(",".join(trainable_layers)))


def _setup_lora_tuning(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    logger.info("Fine-tuning method: LoRA")
    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
            is_mergeable = False

        if is_deepspeed_zero3_enabled():
            assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
            is_mergeable = False


        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path[:-1]
            adapter_to_resume = model_args.adapter_name_or_path[-1]
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        for adapter in adapter_to_merge:
            model: "LoraModel" = PeftModel.from_pretrained(model, adapter, offload_folder=model_args.offload_folder)
            model = model.merge_and_unload()

        if len(adapter_to_merge) > 0:
            logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

        if adapter_to_resume is not None:  # resume lora training
            model = PeftModel.from_pretrained(
                model,
                adapter_to_resume,
                is_trainable=is_trainable,
                offload_folder=model_args.offload_folder,
            )

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
            target_modules = find_all_linear_modules(model)
        else:
            target_modules = finetuning_args.lora_target

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "modules_to_save": finetuning_args.additional_target,
        }


        lora_config = LoraConfig(
            # FIXME: CAUSAL_LM
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_kwargs,
        )
        model = get_peft_model(model, lora_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    if model_args.adapter_name_or_path is not None:
        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model


def init_adapter(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if (not is_trainable) and model_args.adapter_name_or_path is None:
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model

    if finetuning_args.finetuning_type != "lora" and getattr(model, "quantization_method", None):
        raise ValueError("You can only use lora for quantized models.")


    logger.info("Upcasting trainable params to float32.")
    cast_trainable_params_to_fp32 = True

    if is_trainable and finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, model_args, finetuning_args, cast_trainable_params_to_fp32)

    if is_trainable and finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, model_args, finetuning_args, cast_trainable_params_to_fp32)

    if finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            config, model, model_args, finetuning_args, is_trainable, cast_trainable_params_to_fp32
        )

    return model