from dataclasses import dataclass
from typing import Optional
import lightning as L

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import transformers
import tokenizers
import os
import re
import itertools
import functools


# Defaults set for LM1B dataset
@dataclass
class DatasetConfig:
    path: str = "lm1b"
    max_length: int = 128
    batch_size: int = 512  # Please devide by your number of GPUs
    eval_batch_size: int = 64
    num_workers: int = 8
    max_val_samples: Optional[int] = 10000
    wrap: bool = True
    num_proc: int = 24
    streaming: bool = False
    # LM1B typically doesn't use special tokens in wrapped mode
    insert_train_eos: bool = True
    insert_train_special: bool = True
    insert_valid_eos: bool = True
    insert_valid_special: bool = True
    cache_dir: Optional[str] = None


def lm1b_detokenizer(x):
    """LM1B detokenizer from reference dataloaders.py"""
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def _group_texts(examples, block_size, bos, eos, insert_special_tokens=True):
    """Text grouping from reference dataloaders_block.py with optional special tokens"""
    # Concatenate all texts.
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    # We drop the small remainder, and if the total_length < block_size - 2
    # we exclude this batch and return an empty dict.
    if insert_special_tokens:
        new_block_size = block_size - 2  # [BOS] and [EOS] to be added
    else:
        new_block_size = block_size
    total_length = (total_length // new_block_size) * new_block_size
    # Split by chunks of max_len.
    result = {}
    _values = []
    _attn_masks = []
    for i in range(0, total_length, new_block_size):
        if insert_special_tokens:
            _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        else:
            _values.append(concatenated_examples[i : i + new_block_size])
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"] = _values
    result["attention_mask"] = _attn_masks
    return result


class MultiDataset(torch.utils.data.Dataset):
    """
    Multi-dataset loader supporting LM1B and OpenWebText with reference-style preprocessing.
    Migrated from reference dataloaders.py with complete preprocessing pipelines.
    """

    def __init__(
        self,
        tokenizer,
        data_path,
        max_length=1024,
        split="train",
        max_val_samples=None,
        wrap=True,
        num_proc=4,
        streaming=False,
        insert_eos=True,
        insert_special_tokens=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.wrap = wrap
        self.data_path = data_path
        self.split = split
        self.insert_eos = insert_eos
        self.insert_special_tokens = insert_special_tokens

        # Setup tokenizer for wrapping (from reference)
        if wrap:
            self._setup_tokenizer_for_wrapping()

        # Determine detokenizer based on dataset
        if data_path == "lm1b":
            self.detokenizer = lm1b_detokenizer
        else:
            self.detokenizer = None

        # Load dataset using reference logic
        if data_path == "lm1b":
            # Use HuggingFace lm1b dataset
            hf_split = "train" if split == "train" else "test"
            dataset = load_dataset("dvruette/lm1b", split=hf_split)
            if split != "train" and max_val_samples is not None and max_val_samples > 0:
                if not streaming:
                    dataset = dataset.select(range(min(max_val_samples, len(dataset))))
        else:
            if split == "train":
                dataset = load_dataset(data_path, split="train")
            else:  # validation
                dataset = load_dataset(data_path, split="test")
                if max_val_samples is not None and max_val_samples > 0:
                    dataset = dataset.select(range(min(max_val_samples, len(dataset))))

        # Apply preprocessing and tokenization (from reference)
        if streaming:
            tokenized_dataset = dataset.map(self._preprocess_and_tokenize, batched=True, desc="Tokenizing")
        else:
            tokenized_dataset = dataset.map(
                self._preprocess_and_tokenize,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=True,
                desc="Tokenizing",
            )

        # Remove original text columns
        if data_path in ["lm1b", "openwebtext"]:
            tokenized_dataset = tokenized_dataset.remove_columns("text")

        # Group texts if wrapping is enabled
        if wrap:
            EOS = self.tokenizer.encode(self.tokenizer.eos_token)[0]
            BOS = self.tokenizer.encode(self.tokenizer.bos_token)[0]
            group_texts = functools.partial(
                _group_texts, block_size=max_length, bos=BOS, eos=EOS, insert_special_tokens=self.insert_special_tokens
            )

            if streaming:
                chunked_dataset = tokenized_dataset.map(group_texts, batched=True, desc="Grouping")
            else:
                chunked_dataset = tokenized_dataset.map(
                    group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True, desc="Grouping"
                )
            self.dataset = chunked_dataset
        else:
            self.dataset = tokenized_dataset

        # Set format
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def _setup_tokenizer_for_wrapping(self):
        """Setup tokenizer for wrapped batches (from reference)"""
        # Ensure BOS/EOS tokens exist
        if self.tokenizer.bos_token is None:
            if self.tokenizer.cls_token is None:
                raise AttributeError(f"Tokenizer must have a bos_token or cls_token: {self.tokenizer}")
            self.tokenizer.bos_token = self.tokenizer.cls_token
        if self.tokenizer.eos_token is None:
            if self.tokenizer.sep_token is None:
                raise AttributeError(f"Tokenizer must have a eos_token or sep_token: {self.tokenizer}")
            self.tokenizer.eos_token = self.tokenizer.sep_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # GPT2 tokenizer setup from reference
        if isinstance(self.tokenizer, transformers.GPT2TokenizerFast) or isinstance(
            self.tokenizer, transformers.GPT2Tokenizer
        ):
            self.tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
                (self.tokenizer.bos_token, self.tokenizer.bos_token_id),
                (self.tokenizer.eos_token, self.tokenizer.eos_token_id),
            )

    def _apply_detokenizer(self, texts):
        """Apply detokenizer to list of texts"""
        if self.detokenizer is None:
            return texts
        return [self.detokenizer(text) for text in texts]

    def _preprocess_and_tokenize(self, examples):
        """Preprocess and tokenize examples (from reference logic)"""
        texts = examples["text"]

        # Apply detokenizer if available
        if self.detokenizer is not None:
            texts = self._apply_detokenizer(texts)

        # Set tokenizer padding/truncation
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        if self.wrap:
            # Wrapped tokenization: add EOS only if insert_eos=True, BOS added later in group_texts
            tokens = self.tokenizer(
                texts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False
            )
            if self.insert_eos:
                EOS = self.tokenizer.encode(self.tokenizer.eos_token)[0]
                tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
        else:
            # Standard tokenization with truncation/padding
            tokens = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
        return tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataset(config: DatasetConfig, tokenizer, split="train"):
    # Set HF cache if provided
    if cache_dir := config.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = cache_dir

    # Special token insertion control from reference
    if split == "train":
        insert_eos = config.insert_train_eos
        insert_special_tokens = config.insert_train_special
        max_val_samples = None
    else:
        insert_eos = config.insert_valid_eos
        insert_special_tokens = config.insert_valid_special
        max_val_samples = config.max_val_samples

    dataset = MultiDataset(
        tokenizer,
        config.path,
        max_length=config.max_length,
        split=split,
        max_val_samples=max_val_samples,
        wrap=config.wrap,
        num_proc=config.num_proc,
        streaming=config.streaming,
        insert_eos=insert_eos,
        insert_special_tokens=insert_special_tokens,
    )

    return dataset


def get_dataloader(config, tokenizer, split="train"):
    dataset = get_dataset(config, tokenizer, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size if split == "train" else config.eval_batch_size,
        shuffle=(split == "train") and not config.streaming,
        num_workers=config.num_workers,
        pin_memory=True,
    )


# You may use LightningDataModule to wrap the dataset and dataloader
# Or directly use get_dataloader in your training script
class MultiDataModule(L.LightningDataModule):
    def __init__(self, config: DatasetConfig, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def prepare_data(self) -> None:
        _ = get_dataset(self.config, self.tokenizer, split="train")
        _ = get_dataset(self.config, self.tokenizer, split="val")

    def train_dataloader(self):
        return get_dataloader(self.config, self.tokenizer, split="train")

    def val_dataloader(self):
        return get_dataloader(self.config, self.tokenizer, split="val")

    def test_dataloader(self):
        return get_dataloader(self.config, self.tokenizer, split="val")


def _sample_usage():
    # Initialize tokenizer
    num_devices = torch.cuda.device_count()
    cfg = DatasetConfig(batch_size=512 // num_devices)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased",
        trust_remote_code=True,
        cache_dir=cfg.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data module
    datamodule = MultiDataModule(cfg, tokenizer)
    ...
