import os
import time
import yaml
import fire
import mup

import torch
from torch import optim
import torch.distributed.optim
import transformers

import lib.ddp
import lib.utils
import lib.models
import lib.datasets

from plaid_cmt.models.plaid_adapter import PlaidDiffusionAdapter
from plaid_cmt.solvers.teacher_solver import TeacherSolver
from plaid_cmt.losses.consistency import ConsistencyLoss
from plaid_cmt.utils.ema import EMA
from plaid_cmt.utils.lru_cache import LRUCache
from standalone_dataset_fix1 import DatasetConfig, get_dataloader
from torch.nn.parallel import DistributedDataParallel as DDP


def load_plaid_modules(
    weights_dir: str,
    vocab_size: int,
    embed_dim: int,
    dim: int,
    n_blocks: int,
    n_heads: int,
    gamma_0: float,
    gamma_1: float,
    device: torch.device,
):
    """
    Load Plaid modules from a checkpoint directory.

    The directory is expected to contain:
      - noise_schedule.pt
      - gamma_bounds.pt
      - embedding_matrix.pt
      - model.pt

    Args:
        weights_dir: Directory containing checkpoint files.
        vocab_size: Vocabulary size.
        embed_dim: Embedding dimension.
        dim: Hidden dimension for the diffusion model.
        n_blocks: Number of transformer blocks.
        n_heads: Number of attention heads.
        gamma_0: Lower bound for gamma.
        gamma_1: Upper bound for gamma.
        device: Device to load modules onto.
    """
    def create_modules(dim, n_heads):
        return {
            "noise_schedule": lib.models.NoiseSchedule().float(),
            "gamma_bounds": lib.models.GammaBounds(gamma_0, gamma_1).float(),
            "embedding_matrix": lib.models.EmbeddingMatrix(vocab_size, embed_dim).float(),
            "model": lib.models.DiffusionModel(
                dim=dim,
                embed_dim=embed_dim,
                n_blocks=n_blocks,
                n_heads=n_heads,
                vocab_size=vocab_size,
            ).float(),
        }
    
    # Create main modules
    modules = create_modules(dim, n_heads)
    # Create base and delta modules for mup
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    
    # Set mup base shapes before loading weights
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.to(device)
    
    # Load weights after setting base shapes
    for name, module in modules.items():
        ckpt_path = os.path.join(weights_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expected checkpoint file not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        # Ensure loaded weights are float32 to match module dtype
        state_dict = {k: v.float() if v.dtype in (torch.float32, torch.float64) else v 
                     for k, v in state_dict.items()}
        module.load_state_dict(state_dict)
    
    return modules


def main(**kwargs):
    # Distributed initialization is handled by wrap_main, so we can directly
    # use rank() and world_size() which will work after DDP is set up.
    rank = lib.ddp.rank()
    world_size = lib.ddp.world_size()

    # Load config and merge with command-line arguments.
    config_path = kwargs.pop("config_path", "plaid_cmt/configs/cmt_plaid_owt2.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    args = lib.utils.AttributeDict(cfg)
    # Override with command-line arguments (kwargs).
    for k, v in kwargs.items():
        args[k] = v

    # Set defaults for model architecture (can be overridden by config or CLI).
    args.setdefault("dim", 2048)
    args.setdefault("n_blocks", 24)
    args.setdefault("n_heads", 32)
    args.setdefault("gamma_0", -3.0)
    args.setdefault("gamma_1", 6.0)
    args.setdefault("ddim_sampler", False)
    args.setdefault("score_temp", 0.9)
    args.setdefault("cache_size", 1000)

    if rank == 0:
        lib.utils.print_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Use float64 by default as in the original Plaid code.
    torch.set_default_dtype(torch.float64)

    # ----------------- Dataset and Tokenizer -----------------
    # Determine which tokenizer to use based on teacher's vocab_size
    # For CMT, teacher and student MUST use the same embedding space
    use_owt2_tokenizer = getattr(args, "use_owt2_tokenizer", False)
    tokenizer_name = getattr(args, "tokenizer_name", None)
    
    if rank == 0:
        print(f"use_owt2_tokenizer: {use_owt2_tokenizer}, tokenizer_name: {tokenizer_name}")
    
    if use_owt2_tokenizer:
        # Use OWT2 tokenizer (vocab_size=32768) to match teacher
        tokenizer = lib.datasets.openwebtext2_tokenizer()
        vocab = tokenizer.get_vocab()
        word2idx = {k.encode('utf-8'): v for k, v in vocab.items()}
        idx2word = {v: k for k, v in word2idx.items()}
        vocab_size = len(word2idx)
        if rank == 0:
            print(f'Using OWT2 tokenizer, vocab_size: {vocab_size}')
        
        # Use LM1B dataset (matching train_lm1b_bert.py) even with OWT2 tokenizer
        # This allows using OWT2 tokenizer with LM1B data
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        lm1b_path = os.path.expanduser("~/EDLM/datasets/lm1b")
        lm1b_path = os.path.abspath(lm1b_path)
        dataset_config = DatasetConfig(
            path=lm1b_path,
            max_length=args.seq_len,
            batch_size=args.batch_size // world_size,
            eval_batch_size=args.val_batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing device issues
            max_val_samples=10000,
            wrap=True,
            num_proc=24,
            streaming=False,
            insert_train_eos=True,
            insert_train_special=True,
            insert_valid_eos=True,
            insert_valid_special=True,
            cache_dir=None,
        )

        # Create dataloaders with OWT2 tokenizer but LM1B data
        train_dataloader = get_dataloader(dataset_config, tokenizer, split="train")
        val_dataloader = get_dataloader(dataset_config, tokenizer, split="val")
        test_dataloader = get_dataloader(dataset_config, tokenizer, split="val")

        # Create iterators from dataloaders
        def make_iterator(dataloader):
            while True:
                for batch in dataloader:
                    # Extract input_ids from batch (shape: [batch_size, seq_len])
                    yield batch['input_ids'].to(device)

        train_it = make_iterator(train_dataloader)
    else:
        # Use BERT tokenizer (or other HuggingFace tokenizer)
        if tokenizer_name is None:
            tokenizer_name = "google-bert/bert-base-uncased"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        # Ensure pad_token is set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id

        # Create vocab mappings from tokenizer
        vocab = tokenizer.get_vocab()
        word2idx = {k.encode('utf-8'): v for k, v in vocab.items()}
        idx2word = {v: k for k, v in word2idx.items()}
        vocab_size = len(word2idx)
        if rank == 0:
            print(f'Using {tokenizer_name} tokenizer, vocab_size: {vocab_size}')

        # Setup dataset config for LM1B
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        lm1b_path = os.path.expanduser("~/EDLM/datasets/lm1b")
        lm1b_path = os.path.abspath(lm1b_path)
        dataset_config = DatasetConfig(
            path=lm1b_path,
            max_length=args.seq_len,
            batch_size=args.batch_size // world_size,
            eval_batch_size=args.val_batch_size,
            num_workers=0,  # Set to 0 to avoid multiprocessing device issues
            max_val_samples=10000,
            wrap=True,
            num_proc=24,
            streaming=False,
            insert_train_eos=True,
            insert_train_special=True,
            insert_valid_eos=True,
            insert_valid_special=True,
            cache_dir=None,
        )

        # Create dataloaders
        train_dataloader = get_dataloader(dataset_config, tokenizer, split="train")
        val_dataloader = get_dataloader(dataset_config, tokenizer, split="val")
        test_dataloader = get_dataloader(dataset_config, tokenizer, split="val")

        # Create iterators from dataloaders
        def make_iterator(dataloader):
            while True:
                for batch in dataloader:
                    # Extract input_ids from batch (shape: [batch_size, seq_len])
                    yield batch['input_ids'].to(device)

        train_it = make_iterator(train_dataloader)

    # ----------------- Teacher / Student modules -----------------
    # Use vocab_size from tokenizer instead of config
    embed_dim = args.embed_dim

    # Load teacher modules from checkpoints.
    # IMPORTANT: Teacher and student must use the same vocab_size for CMT to work.
    # If teacher was trained with a different tokenizer, you need to either:
    # 1. Use the same tokenizer for student, or
    # 2. Retrain teacher with the same tokenizer
    # For now, we assume teacher and student use the same vocab_size
    teacher_modules = load_plaid_modules(
        args.teacher_weights_dir,
        vocab_size,  # Must match teacher's vocab_size
        embed_dim,
        args.dim,
        args.n_blocks,
        args.n_heads,
        args.gamma_0,
        args.gamma_1,
        device,
    )
    # Initialize student from the same weights for CMT.
    student_modules = load_plaid_modules(
        args.teacher_weights_dir,
        vocab_size,  # Same vocab_size as teacher
        embed_dim,
        args.dim,
        args.n_blocks,
        args.n_heads,
        args.gamma_0,
        args.gamma_1,
        device,
    )

    teacher_adapter = PlaidDiffusionAdapter(teacher_modules, vocab_size, embed_dim, device)
    student_adapter = PlaidDiffusionAdapter(student_modules, vocab_size, embed_dim, device)

    # Wrap student model with DDP for multi-GPU training
    # Only wrap the diffusion model, not the entire adapter
    student_model_ddp = DDP(
        student_adapter.model,
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    # Replace the model in student_adapter with DDP-wrapped version
    student_adapter.model = student_model_ddp

    # EMA of the student model serves as the teacher for CMT.
    # Use .module to access underlying model from DDP wrapper
    ema = EMA(student_adapter.model.module, ema_decay=args.ema_decay)

    noise_schedule = teacher_modules["noise_schedule"]
    gamma_bounds = teacher_modules["gamma_bounds"]

    # Create in-process LRU cache for teacher outputs.
    cache = LRUCache(max_size=args.cache_size)

    teacher_solver = TeacherSolver(
        adapter=teacher_adapter,
        noise_schedule=noise_schedule,
        gamma_bounds=gamma_bounds,
        n_steps=args.teacher_steps,
        cache=cache,
        ddim_sampler=args.ddim_sampler,
        score_temp=args.score_temp,
    )

    # ----------------- Optimizer / Loss -----------------
    # Use ZeroRedundancyOptimizer for distributed training (like train_lm1b_bert.py)
    optimizer = torch.distributed.optim.ZeroRedundancyOptimizer(
        student_adapter.parameters(),
        optimizer_class=optim.AdamW,
        parameters_as_bucket_view=True,
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    consistency_loss = ConsistencyLoss(weight=args.cmt_weight)

    # ----------------- Training loop -----------------
    global_step = 0
    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    student_adapter.train()

    while global_step < args.max_steps:
        batch = next(train_it)  # [B, L]
        batch = batch.to(device)

        global_step += 1

        # Encode tokens to continuous embeddings x0.
        with torch.no_grad():
            x0 = teacher_adapter.encode(batch)  # [B, L, D]

        # Sample a time t in [t_min, t_max].
        t = torch.empty((), device=device).uniform_(args.t_min, args.t_max)

        # Forward diffusion: x_t = alpha(t) * x0 + sigma(t) * epsilon.
        gamma_0, gamma_1 = gamma_bounds()
        gamma_tilde = noise_schedule(t[None]).double()  # [1]
        gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_tilde  # [1]
        alpha_sq = torch.sigmoid(-gamma_t)
        sigma_sq = torch.sigmoid(gamma_t)
        alpha_t = alpha_sq.sqrt()
        sigma_t = sigma_sq.sqrt()

        # Ensure x0 is float32 (matching embedding matrix dtype)
        x0 = x0.float()
        # Ensure eps and x_t are float32 to match model weights
        eps = torch.randn_like(x0, dtype=torch.float32)
        x_t = (alpha_t.view(1, 1, 1) * x0 + sigma_t.view(1, 1, 1) * eps).float()

        # Teacher multi-step trajectory to approximate x0.
        # Use a simple cache key based on batch content hash (for simplicity,
        # we use the first token of each sequence as a rough identifier).
        # In practice, you might want a more sophisticated cache key.
        cache_key = None  # Disable caching for now; can enable with proper keys
        with torch.no_grad():
            x0_teacher = teacher_solver(
                x_t, t_scalar=float(t.item()), input_ids=batch, cache_key=cache_key
            )

        # Student single-step prediction at time t.
        # Note: DiffusionModel handles mixed precision internally (bf16 for transformer blocks only).
        # We don't use autocast here to avoid type mismatches with float32/float64 parts.
        # Ensure x_selfcond is float32 to match model weights (modules are created with .float())
        x_selfcond = torch.zeros_like(x_t, dtype=torch.float32)
        logits_s, x_reconst_s = student_adapter(
            z_t=x_t,
            gamma_t=gamma_t,
            input_ids=batch,
            x_selfcond=x_selfcond,
            selfcond_mask=None,
        )
        x0_student = x_reconst_s

        loss_cmt = consistency_loss(x0_teacher, x0_student)

        optimizer.zero_grad()
        loss_cmt.backward()
        torch.nn.utils.clip_grad_norm_(student_adapter.parameters(), args.grad_clip)
        optimizer.step()

        # EMA update on the diffusion model only.
        # Use .module to access underlying model from DDP wrapper
        ema.update(student_adapter.model.module)

        if rank == 0 and (global_step % args.log_every == 0):
            print(f"[CMT] step={global_step} loss_cmt={loss_cmt.item():.6f}")

        if global_step % args.ckpt_every == 0:
            # For ZeroRedundancyOptimizer, need to consolidate state on all ranks
            # before saving on rank 0
            optimizer.consolidate_state_dict(to=0)
            if rank == 0:
                # Save checkpoint: use .module to get underlying model state from DDP
                ckpt = {
                    "student": student_adapter.state_dict(),
                    "ema_model": ema.ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "config": cfg,
                }
                torch.save(ckpt, os.path.join(args.out_dir, f"cmt_plaid_step_{global_step}.pt"))

    # Cleanup is handled automatically by wrap_main


if __name__ == "__main__":
    fire.Fire(lib.ddp.wrap_main(main))


