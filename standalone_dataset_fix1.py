from dataclasses import dataclass
from typing import Optional

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
    # Create attention mask as a list (not tensor) to avoid CUDA issues in multiprocessing
    # The dataset will convert to tensors later when set_format is called
    attn_mask_list = [1] * block_size
    for i in range(0, total_length, new_block_size):
        if insert_special_tokens:
            _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        else:
            _values.append(concatenated_examples[i : i + new_block_size])
        # _attn_masks.append(torch.ones(block_size))
        _attn_masks.append(attn_mask_list.copy())
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
        # Check if it's LM1B (either "lm1b" string or local path containing "lm1b")
        is_lm1b = (data_path == "lm1b" or 
                   (isinstance(data_path, str) and os.path.isdir(data_path) and "lm1b" in data_path.lower()))
        
        if is_lm1b:
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
        elif is_lm1b and os.path.isdir(data_path):
            # Use local LM1B dataset directory
            hf_split = "train" if split == "train" else "test"
            # Try loading from local directory - first try as a local dataset path
            try:
                # Try loading as a local dataset (if it's a saved dataset)
                from datasets import load_from_disk
                dataset = load_from_disk(data_path)
                if hf_split in dataset:
                    dataset = dataset[hf_split]
                elif split == "train" and "train" in dataset:
                    dataset = dataset["train"]
                elif split != "train" and "test" in dataset:
                    dataset = dataset["test"]
            except:
                # If load_from_disk fails, try loading with HuggingFace dataset loader
                try:
                    dataset = load_dataset("dvruette/lm1b", split=hf_split, data_dir=data_path)
                except:
                    # Last resort: try loading the directory as a dataset
                    dataset = load_dataset(data_path, split=hf_split)
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
        if is_lm1b or data_path in ["lm1b", "openwebtext"]:
            if "text" in tokenized_dataset.column_names:
                tokenized_dataset = tokenized_dataset.remove_columns("text")

        # Group texts if wrapping is enabled
        if wrap:
            # Handle both tokenizers.Tokenizer (OWT2) and HuggingFace tokenizers
            from tokenizers import Tokenizer as TokenizersTokenizer
            if isinstance(self.tokenizer, TokenizersTokenizer):
                # For OWT2 tokenizer, use the token IDs we set up
                EOS = self._eos_token_id
                BOS = self._bos_token_id
            else:
                # For HuggingFace tokenizers
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
        # Check if this is a tokenizers.Tokenizer (OWT2) or HuggingFace tokenizer
        from tokenizers import Tokenizer as TokenizersTokenizer
        is_tokenizers_lib = isinstance(self.tokenizer, TokenizersTokenizer)
        
        if is_tokenizers_lib:
            # For tokenizers.Tokenizer (OWT2), we need to set up BOS/EOS token IDs
            # OWT2 tokenizer typically uses token ID 0 for EOS
            vocab = self.tokenizer.get_vocab()
            # Try to find EOS token (usually token ID 0 or a special token)
            if 0 in vocab.values():
                # Find the token string for ID 0
                for token_str, token_id in vocab.items():
                    if token_id == 0:
                        self._eos_token_str = token_str.decode('utf-8') if isinstance(token_str, bytes) else token_str
                        self._eos_token_id = 0
                        break
            else:
                # Fallback: use token ID 0 as EOS
                self._eos_token_str = None
                self._eos_token_id = 0
            
            # BOS token is typically not used in OWT2, but we'll use EOS as BOS for wrapping
            self._bos_token_str = self._eos_token_str
            self._bos_token_id = self._eos_token_id
        else:
            # For HuggingFace transformers tokenizers
            # Increase model_max_length to allow longer sequences during tokenization
            # Since we'll chunk sequences later, we don't need to truncate here
            if hasattr(self.tokenizer, 'model_max_length'):
                # Set to a very large value to avoid truncation warnings
                # The actual chunking will happen in _group_texts
                self.tokenizer.model_max_length = 1_000_000  # Large enough for most texts
            
            # Ensure BOS/EOS tokens exist
            if not hasattr(self.tokenizer, 'bos_token') or self.tokenizer.bos_token is None:
                if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token is not None:
                    self.tokenizer.bos_token = self.tokenizer.cls_token
                else:
                    raise AttributeError(f"Tokenizer must have a bos_token or cls_token: {self.tokenizer}")
            if not hasattr(self.tokenizer, 'eos_token') or self.tokenizer.eos_token is None:
                if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
                    self.tokenizer.eos_token = self.tokenizer.sep_token
                else:
                    raise AttributeError(f"Tokenizer must have a eos_token or sep_token: {self.tokenizer}")
            if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
                # Try to use eos_token as pad_token first
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                else:
                    # Add a new pad token if eos_token is also not available
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    self.tokenizer.pad_token_id = self.tokenizer.get_vocab()["[PAD]"]

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

        # Check tokenizer type
        from tokenizers import Tokenizer as TokenizersTokenizer
        is_tokenizers_lib = isinstance(self.tokenizer, TokenizersTokenizer)

        # Set tokenizer padding/truncation (only for HuggingFace tokenizers)
        if not is_tokenizers_lib:
            if hasattr(self.tokenizer, 'padding_side'):
                self.tokenizer.padding_side = "right"
            if hasattr(self.tokenizer, 'truncation_side'):
                self.tokenizer.truncation_side = "right"

        if self.wrap:
            # Wrapped tokenization: add EOS only if insert_eos=True, BOS added later in group_texts
            # Explicitly disable truncation since we'll chunk sequences later
            if is_tokenizers_lib:
                # For tokenizers.Tokenizer (OWT2), use encode_batch
                tokenized = self.tokenizer.encode_batch(texts)
                tokens = {"input_ids": [enc.ids for enc in tokenized]}
            else:
                # For HuggingFace tokenizers
                tokens = self.tokenizer(
                    texts, 
                    add_special_tokens=False, 
                    return_attention_mask=False, 
                    return_token_type_ids=False,
                    truncation=False,  # Don't truncate - we'll chunk later
                    padding=False,  # Don't pad - we'll handle this in grouping
                )
            
            if self.insert_eos:
                # Handle both tokenizers.Tokenizer (OWT2) and HuggingFace tokenizers
                if is_tokenizers_lib:
                    EOS = self._eos_token_id
                else:
                    EOS = self.tokenizer.encode(self.tokenizer.eos_token)[0]
                tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
        else:
            # Standard tokenization with truncation/padding
            if is_tokenizers_lib:
                # For tokenizers.Tokenizer (OWT2), encode and pad/truncate manually
                tokenized = self.tokenizer.encode_batch(texts, add_special_tokens=True)
                input_ids = []
                attention_mask = []
                for enc in tokenized:
                    ids = enc.ids[:self.max_length]  # Truncate
                    # Pad to max_length
                    pad_length = self.max_length - len(ids)
                    ids = ids + [0] * pad_length  # Use 0 as pad token ID
                    input_ids.append(ids)
                    attention_mask.append([1] * (self.max_length - pad_length) + [0] * pad_length)
                tokens = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            else:
                # For HuggingFace tokenizers
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
    # Don't provide a generator - let DataLoader create its own internally
    # This avoids device mismatch issues when torch.set_default_device('cuda') is set
    # The DataLoader will handle generator creation with the correct device
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size if split == "train" else config.eval_batch_size,
        shuffle=(split == "train") and not config.streaming,
        num_workers=config.num_workers,
        pin_memory=True
    )


# You may use LightningDataModule to wrap the dataset and dataloader
# Or directly use get_dataloader in your training script
class MultiDataModule:
    """
    LightningDataModule wrapper (optional).
    Lightning import is delayed to avoid import errors if Lightning is not available.
    """
    def __init__(self, config: DatasetConfig, tokenizer):
        # Lazy import to avoid import errors if Lightning is not available
        try:
            import lightning as L
            self._L = L
            # Create a temporary class that inherits from LightningDataModule
            class _LightningDataModule(L.LightningDataModule):
                def __init__(self, config, tokenizer):
                    super().__init__()
                    self.config = config
                    self.tokenizer = tokenizer
            # Use composition instead of inheritance to avoid import issues
            self._lightning_module = _LightningDataModule(config, tokenizer)
        except ImportError:
            raise ImportError(
                "Lightning is required for MultiDataModule. "
                "Use get_dataloader() directly instead, or install lightning."
            )
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
