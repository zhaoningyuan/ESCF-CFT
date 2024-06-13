from .utils.callbacks import LogCallback
from .utils.logger import get_logger
from .hparams.parser import get_train_args
from .sft.workflow import run_sft


from typing import TYPE_CHECKING, List, Optional, Dict, Any
if TYPE_CHECKING:
    from transformers import TrainerCallback


logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks.append(LogCallback(training_args.output_dir))

    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)



