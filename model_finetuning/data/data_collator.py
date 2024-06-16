from transformers import DataCollatorForSeq2Seq

class CustomDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    r"""
    Customized data collator for seq2seq tasks.
    """
    
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
    
    def __call__(self, features, **kwargs):
        r"""
        Collate `features` into batch.
        """
        batch = super().__call__(features, **kwargs)
        ## Remove labels from batch if not present
        if "labels" in batch and batch["labels"] is None:
            batch.pop("labels", None)
        return batch