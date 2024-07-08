import os
import random
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled
from transformers.utils.versions import require_version

from ...utils.logger import get_logger
from ...utils.misc import get_current_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer

    from ...hparams.args import ModelArguments


logger = get_logger(__name__)

def configure_quantization(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: Dict[str, Any],
) -> None:
    r"""
    Priority: PTQ-quantized (training) > AutoGPTQ (export) > Bitsandbytes (training)
    """
    if model_args.quantization_bit is not None:  # on-the-fly
            if model_args.quantization_bit == 8:
                require_version("bitsandbytes>=0.37.0", "To fix: pip install bitsandbytes>=0.37.0")
                init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif model_args.quantization_bit == 4:
                require_version("bitsandbytes>=0.39.0", "To fix: pip install bitsandbytes>=0.39.0")
                init_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=model_args.compute_dtype,
                    bnb_4bit_use_double_quant=model_args.double_quantization,
                    bnb_4bit_quant_type=model_args.quantization_type,
                    bnb_4bit_quant_storage=model_args.compute_dtype,  # crucial for fsdp+qlora
                )
            else:
                raise ValueError("Bitsandbytes only accepts 4-bit or 8-bit quantization.")

            # Do not assign device map if:
            # 1. deepspeed zero3 or fsdp (train)
            # 2. auto quantization device map (inference)
            if is_deepspeed_zero3_enabled() or is_fsdp_enabled() or model_args.quantization_device_map == "auto":
                if model_args.quantization_bit != 4:
                    raise ValueError("Only 4-bit quantized model can use fsdp+qlora or auto device map.")

                require_version("bitsandbytes>=0.43.0", "To fix: pip install bitsandbytes>=0.43.0")
            else:
                init_kwargs["device_map"] = {"": get_current_device()}  # change auto device map for inference

            logger.info("Quantizing model to {} bit with bitsandbytes.".format(model_args.quantization_bit))
