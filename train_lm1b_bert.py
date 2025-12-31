import contextlib
import fire
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mup
import numpy as np
import lib.ddp
import lib.decay_to_init
import lib.ema
import lib.models
import lib.ops
import lib.utils
import os
import random
import time
import torch
import torch.distributed.optim
import torch.nn.functional as F
import tqdm
import transformers
from torch import nn, optim, autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from standalone_dataset_fix1 import DatasetConfig, get_dataloader

def count_non_embedding_params(modules):
    """Count parameters excluding embedding_matrix"""
    total = 0
    for name, module in modules.items():
        if name != 'embedding_matrix':
            for param in module.parameters():
                total += param.numel()
    return total

def main(**args):
    args = lib.utils.AttributeDict(args)
    # Reduce default batch_size to avoid OOM with 63M parameter model
    args.setdefault('batch_size', 256)
    # args.setdefault('dataset', 'openwebtext2')
    # Increase grad_accum_steps to maintain effective batch size while reducing memory
    args.setdefault('grad_accum_steps', 1)
    args.setdefault('hook_freq', 10000)
    args.setdefault('lr', 1.4e-3)
    args.setdefault('lr_warmup_steps', 2500)
    args.setdefault('bias_warmup_steps', 5000)
    args.setdefault('lr_decay', True)
    args.setdefault('print_freq', 1000)
    args.setdefault('save_weights', True)
    # args.setdefault('steps', 92000)
    args.setdefault('steps', 1_000_000)
    args.setdefault('weights_path', None)
    args.setdefault('reconst_weight', 1.0)
    # BERT version model config - fixed parameters
    args.setdefault('dim', 768)
    args.setdefault('n_blocks', 12)
    args.setdefault('n_heads', 12)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    # args.setdefault('seq_len', 256)
    args.setdefault('seq_len', 128)
    args.setdefault('val_steps', 100)
    args.setdefault('val_batch_size', 64)
    args.setdefault('weight_decay', 4e-5)
    args.setdefault('first_step', 0)
    args.setdefault('auto_resume', False)
    args.setdefault('decay_to_init', 0.)
    args.setdefault('ema', 0.)
    args.setdefault('beta1', 0.9)
    args.setdefault('beta2', 0.99)
    args.setdefault('selfcond', True)
    args.setdefault('n_short_seqs', 2)
    args.setdefault('clip_quantile', 0.95)
    args.setdefault('reconst_bs_ema', 0.997)
    args.setdefault('final_val_steps', 3000)
    args.setdefault('tokenizer_name', 'google-bert/bert-base-uncased')  # BERT tokenizer for LM1B
    args.setdefault('checkpoint_dir', 'checkpoints_bert')  # Checkpoint directory to distinguish from GPT2 version

    lib.utils.print_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Don't set default device - explicitly specify device for all tensors

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)
    
    # Set device for CUDA operations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize BERT tokenizer for LM1B
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
    )
    # BERT tokenizer typically has pad_token, but ensure it's set
    # This will be handled properly in _setup_tokenizer_for_wrapping
    # but we set it here to avoid warnings during tokenizer initialization
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.unk_token is not None:
            # BERT may use unk_token, but pad_token should be set
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            # Will be set in _setup_tokenizer_for_wrapping
            pass
    
    # Create vocab mappings from tokenizer
    vocab = tokenizer.get_vocab()
    word2idx = {k.encode('utf-8'): v for k, v in vocab.items()}
    idx2word = {v: k for k, v in word2idx.items()}
    vocab_size = len(word2idx)
    print(f'vocab_size: {vocab_size}')

    # Setup dataset config for LM1B
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # Expand ~ in path and convert to absolute path
    lm1b_path = os.path.expanduser("~/EDLM/datasets/lm1b")
    lm1b_path = os.path.abspath(lm1b_path)
    dataset_config = DatasetConfig(
        path=lm1b_path,
        max_length=args.seq_len,
        batch_size=args.batch_size // num_devices,
        eval_batch_size=args.val_batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing device issues with CUDA default device
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
    # Note: num_workers=0 to avoid multiprocessing device issues with CUDA default device
    train_dataloader = get_dataloader(dataset_config, tokenizer, split="train")
    val_dataloader = get_dataloader(dataset_config, tokenizer, split="val")
    test_dataloader = get_dataloader(dataset_config, tokenizer, split="val")

    # Create iterators from dataloaders
    def make_iterator(dataloader):
        while True:
            for batch in dataloader:
                # Extract input_ids from batch (shape: [batch_size, seq_len])
                # Explicitly move to device
                yield batch['input_ids'].to(device)

    train_iterator = make_iterator(train_dataloader)
    val_iterator = make_iterator(val_dataloader)
    test_iterator = make_iterator(test_dataloader)

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, vocab_size).float()
        }
    
    # BERT version uses fixed model configuration
    # Verify that dim is divisible by n_heads
    if args.dim % args.n_heads != 0:
        raise ValueError(f'dim ({args.dim}) must be divisible by n_heads ({args.n_heads})')
    
    # Create test modules to count parameters
    test_modules = create_modules(args.dim, args.n_heads)
    actual_params = count_non_embedding_params(test_modules)
    print(f'BERT model config: dim={args.dim}, n_blocks={args.n_blocks}, n_heads={args.n_heads}, seq_len={args.seq_len}')
    print(f'😄Non-embedding parameters: {actual_params:,}')
    for name, module in test_modules.items():
        if name == 'embedding_matrix':
            for param in module.parameters():
                print(f'😄Embedding parameters: {param.numel()}')
    
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.to(device)
        print(key+':')
        lib.utils.print_model(main)
    
    # Print final non-embedding parameter count
    final_non_embedding = count_non_embedding_params(modules)
    print(f'\nFinal non-embedding parameters: {final_non_embedding:,}')

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = args.checkpoint_dir
    if lib.ddp.rank() == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    # Synchronize all processes to ensure directory is created
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    def load_weights(weights_path):
        print(f'Loading weights from {weights_path}')
        for name, module in modules.items():
            weight_file = os.path.join(weights_path, f'{name}.pt')
            if os.path.exists(weight_file):
                module.load_state_dict(torch.load(
                    weight_file,
                    map_location=device
                ))
            else:
                print(f'Warning: Weight file not found: {weight_file}')

    if args.auto_resume:
        assert(args.save_weights)

    first_step = args.first_step
    step_file = os.path.join(checkpoint_dir, 'step')
    # Check if any model checkpoint exists (e.g., model.pt, noise_schedule.pt, etc.)
    model_file = os.path.join(checkpoint_dir, 'model.pt')
    
    if args.auto_resume and os.path.exists(step_file):
        # Check if at least one model file exists
        checkpoint_exists = any(
            os.path.exists(os.path.join(checkpoint_dir, f'{name}.pt'))
            for name in modules.keys()
        )
        if checkpoint_exists:
            load_weights(checkpoint_dir)
            with open(step_file, 'r') as f:
                first_step = int(f.read()) + 1
            print(f'Resumed from checkpoint at step {first_step - 1} in {checkpoint_dir}/')
        else:
            print(f'Warning: step file exists but no model checkpoints found in {checkpoint_dir}/')
    elif args.weights_path is not None:
        load_weights(args.weights_path)

    print(f'Starting from step {first_step}')

    ddp_modules = {
        name: DDP(module, broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )
        for name, module in modules.items()
    }

    print('DDP initialized')

    emas = {
        name: lib.ema.EMA(module, args.ema)
        for name, module in modules.items()
    }

    decay_to_init = {
        name: lib.decay_to_init.DecayToInit(module, args.decay_to_init)
        for name, module in modules.items()
    }

    loss_ema_bias     = torch.tensor(1e-8, device=device)
    reconst_ema       = torch.tensor(1e-8, device=device)
    diffusion_ema     = torch.tensor(1e-8, device=device)
    reconst_sqr_ema   = torch.tensor(1e-8, device=device)
    diffusion_sqr_ema = torch.tensor(1e-8, device=device)
    reconst_bs_cache  = {}
    def forward(step=None, accum_step=None, accum_total=None, x_eval=None):
        """
        Train mode: step, accum_step, accum_total
        Eval mode: x_eval
        """
        nonlocal reconst_ema, diffusion_ema, reconst_sqr_ema, diffusion_sqr_ema

        train_mode = (x_eval is None)
        if train_mode:
            x = next(train_iterator)
            batch_size = x.shape[0] * accum_total
            if step not in reconst_bs_cache:
                # Synchronize EMA vars
                reconst_ema       = lib.ddp.reduce_mean(reconst_ema)
                reconst_sqr_ema   = lib.ddp.reduce_mean(reconst_sqr_ema)
                diffusion_ema     = lib.ddp.reduce_mean(diffusion_ema)
                diffusion_sqr_ema = lib.ddp.reduce_mean(diffusion_sqr_ema)
                # Compute reconst_bs
                b = 1 / loss_ema_bias # Bias correction factor
                reconst_std   = (b*reconst_sqr_ema   - (b*reconst_ema)**2).clamp(min=0).sqrt()
                diffusion_std = (b*diffusion_sqr_ema - (b*diffusion_ema)**2).clamp(min=0).sqrt()
                reconst_bs = batch_size * (reconst_std / (1e-8 + reconst_std + diffusion_std))
                reconst_bs = int(reconst_bs.round().clamp(1, batch_size-1))
                reconst_bs_cache[step] = reconst_bs
            reconst_bs = reconst_bs_cache[step]
            avg_reconst_bs = float(reconst_bs)
        else:
            x = x_eval
            batch_size = x.shape[0]
            reconst_bs = (batch_size // 8)
            reconst_bs += int(np.random.binomial(1, (batch_size % 8) / 8.))
            avg_reconst_bs = batch_size / 8.

        embedding_matrix = ddp_modules['embedding_matrix']()

        selfcond_mask = torch.zeros([batch_size], device=device)
        avg_selfcond_mask = 0.
        if args.selfcond:
            if train_mode:
                offset = int(np.random.randint(4))
                selfcond_mask[offset::4].add_(1)
                avg_selfcond_mask = 0.25
            else:
                selfcond_mask.add_(1)
                avg_selfcond_mask = 1.

        t = torch.empty([batch_size], device=device)
        # First entries of t are used for reconst_loss below
        t[:reconst_bs] = 0
        # Low-discrepancy sampler for the remaining entries of t
        t[reconst_bs:] = torch.arange(
            batch_size - reconst_bs, device=device)
        if train_mode:
            t[reconst_bs:] += float(np.random.RandomState(step).uniform())
        else:
            t[reconst_bs:] += float(np.random.uniform())
        t[reconst_bs:] /= batch_size - reconst_bs
        t.requires_grad = True

        if train_mode:
            batch_size //= accum_total
            selfcond_mask = selfcond_mask.chunk(accum_total)[accum_step]
            t = t.chunk(accum_total)[accum_step]
            reconst_bs = int(t.eq(0).sum())
            avg_reconst_bs /= accum_total

        selfcond_idx = selfcond_mask.nonzero()[:,0]

        with torch.enable_grad():
            # Don't propagate grads for the first reconst_bs entries of t
            gamma = torch.cat([
                ddp_modules['noise_schedule'](t[:reconst_bs]).detach(),
                ddp_modules['noise_schedule'](t[reconst_bs:])
            ])
            gamma_prime = autograd.grad(gamma.sum(), [t], create_graph=True)[0]
        # Edits gradients so that the noise schedule minimizes
        # E[loss^2] while the rest of the model minimizes E[loss].
        def set_grad_hook(tensor):
            if tensor.requires_grad:
                def grad_hook(grad):
                    handle.remove()
                    new_grad = torch.clone(grad.detach())
                    new_grad[reconst_bs:] *= 2. * (
                        grad_hook_loss[reconst_bs:].detach()
                    )
                    return new_grad
                handle = tensor.register_hook(grad_hook)
        gamma = gamma.clone()
        set_grad_hook(gamma)
        set_grad_hook(gamma_prime)
        gamma_0, gamma_1 = ddp_modules['gamma_bounds']()
        gamma = gamma_0 + (gamma_1 - gamma_0) * gamma
        gamma_prime = (gamma_1 - gamma_0) * gamma_prime

        gamma = torch.lerp(gamma, gamma.detach(), selfcond_mask)
        gamma_prime = torch.lerp(gamma_prime, gamma_prime.detach(), selfcond_mask)

        # Quantities derived from gamma, gamma_prime, gamma_1:
        alpha_squared = torch.sigmoid(-gamma)
        sigma_squared = torch.sigmoid(gamma)
        alpha = alpha_squared.sqrt()
        sigma = sigma_squared.sqrt()
        snr_prime = -(-gamma).exp() * gamma_prime # SNR = exp(-gamma)
        alpha_1 = torch.sigmoid(-gamma_1).sqrt()
        sigma_1 = torch.sigmoid(gamma_1).sqrt()

        # Construct z (with reparam. trick gradients)
        x_embed = embedding_matrix[x]
        x_embed = torch.lerp(x_embed, x_embed.detach(), selfcond_mask.float()[:,None,None])
        z = torch.randn(
            [x.shape[0], x.shape[1], args.embed_dim],
            dtype=torch.float32, device=device
        )
        z.mul_(sigma[:,None,None])
        z.add_(alpha[:,None,None] * x_embed)

        cu_seqlens = None
        cu_seqlens_selfcond = None
        if train_mode:
            accum_interval = max(accum_total // args.n_short_seqs, 1)
            accum_offset = int(np.random.RandomState(step).randint(accum_interval))
            accum_n = args.n_short_seqs * accum_interval // accum_total
            if accum_step % accum_interval == accum_offset:
                seqlens = torch.zeros([batch_size, 2], device=device, dtype=torch.int64)
                seqlens[:,0] = x.shape[1]
                positions = torch.randperm(batch_size, device=device)[:accum_n]
                lens = torch.randint(1, x.shape[1], [accum_n], device=device)
                seqlens[positions, 0] = lens
                seqlens[positions, 1] = x.shape[1] - lens
                cu_seqlens = torch.zeros([seqlens.numel()+1], dtype=torch.int32, device=device)
                cu_seqlens[1:] = seqlens.view(-1).cumsum(dim=0)
                cu_seqlens_selfcond = torch.zeros([seqlens[selfcond_idx].numel()+1], dtype=torch.int32, device=device)
                cu_seqlens_selfcond[1:] = seqlens[selfcond_idx].view(-1).cumsum(dim=0)

        if train_mode:
            bias_scale = min(1., (step + 1e-8) / (args.bias_warmup_steps + 1e-8))
        else:
            bias_scale = 1.

        # Model forward pass for self-conditioning
        x_selfcond = torch.zeros_like(z)
        if len(selfcond_idx) > 0:
            with torch.no_grad():
                z_selfcond = z[selfcond_idx]
                gamma_selfcond = gamma[selfcond_idx]
                logits, x_reconst = ddp_modules['model'](
                    z_selfcond, gamma_selfcond, embedding_matrix, bias_scale,
                    torch.zeros_like(z_selfcond),
                    cu_seqlens=cu_seqlens_selfcond
                )
                del logits
                x_selfcond[selfcond_idx] = x_reconst

        # Main model forward pass
        with torch.enable_grad():
            logits, x_reconst = ddp_modules['model'](
                z, gamma, embedding_matrix, bias_scale, x_selfcond,
                selfcond_mask=selfcond_mask,
                cu_seqlens=cu_seqlens
            )

        # Loss terms
        reconst_loss = lib.ops.cross_entropy(
            logits[:reconst_bs],
            x[:reconst_bs]
        ).mean(dim=1).double()

        alpha_1_masked = torch.lerp(alpha_1, alpha_1.detach(), selfcond_mask)[:,None,None]
        sigma_1_masked = torch.lerp(sigma_1, sigma_1.detach(), selfcond_mask)[:,None,None]
        prior_loss = lib.ops.gaussian_kl(
            (alpha_1_masked * x_embed),
            sigma_1_masked,
            torch.tensor(0., device=device),
            torch.tensor(1., device=device)
        ).sum(dim=2).mean()

        diffusion_loss = (x_embed - x_reconst).pow(2)
        diffusion_loss = diffusion_loss.mean(dim=1).double().sum(dim=1)
        diffusion_loss = -0.5*(snr_prime * diffusion_loss)

        if train_mode:
            with torch.no_grad():
                loss_ema_bias.lerp_(     torch.tensor(1., device=device),                                                   1 - args.reconst_bs_ema)
                reconst_ema.lerp_(       (args.reconst_weight * reconst_loss).sum()        / avg_reconst_bs,                1 - args.reconst_bs_ema)
                reconst_sqr_ema.lerp_(   (args.reconst_weight * reconst_loss).pow(2).sum() / avg_reconst_bs,                1 - args.reconst_bs_ema)
                diffusion_ema.lerp_(     diffusion_loss[reconst_bs:].sum()                 / (batch_size - avg_reconst_bs), 1 - args.reconst_bs_ema)
                diffusion_sqr_ema.lerp_( diffusion_loss[reconst_bs:].pow(2).sum()          / (batch_size - avg_reconst_bs), 1 - args.reconst_bs_ema)

        grad_hook_loss = diffusion_loss # Used above (weird variable scope)

        loss = (args.reconst_weight * reconst_loss).sum() / avg_reconst_bs
        loss += diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
        loss += prior_loss

        if args.selfcond:
            nll = (reconst_loss * selfcond_mask[:reconst_bs]).sum() / (avg_reconst_bs * avg_selfcond_mask)
            nll += (diffusion_loss[reconst_bs:] * selfcond_mask[reconst_bs:]).sum() / ((batch_size - avg_reconst_bs) * avg_selfcond_mask)
            nll += prior_loss
        else:
            nll = reconst_loss.sum() / avg_reconst_bs
            nll += diffusion_loss[reconst_bs:].sum() / (batch_size - avg_reconst_bs)
            nll += prior_loss

        return (
            loss,
            nll,
            reconst_loss.sum() / avg_reconst_bs,
            prior_loss,
            gamma_0,
            gamma_1,
            torch.tensor(reconst_bs, device=device),
        )

    learning_rates = {
        'model': args.lr,
        'noise_schedule': 1e-2,
        'gamma_bounds': 1e-2,
        'embedding_matrix': 1e-2,
    }

    weight_decays = {
        'model': args.weight_decay,
        'noise_schedule': 0.,
        'gamma_bounds': 1e-3,
        'embedding_matrix': 0.,
    }

    def optimizer_impl(param_groups, **kwargs):
        assert('weight_decay' not in kwargs)
        modules_seen = set()
        for i, param_group in enumerate(param_groups):
            weight_decay_set = False
            for name in modules:
                group_params = param_group['params']
                module_params = list(modules[name].parameters())
                if all([any([p is p2 for p2 in module_params]) for p in group_params]):
                    assert(not weight_decay_set)
                    assert(param_group['weight_decay'] == 0.)
                    param_group['weight_decay'] = (
                        weight_decays[name] / (param_group['lr']+1e-16)
                    )
                    weight_decay_set = True
                    modules_seen.add(name)
            assert(weight_decay_set)
        assert(all([name in modules_seen for name in modules]))

        return torch.distributed.optim.ZeroRedundancyOptimizer(param_groups,
            optimizer_class=optim.AdamW, parameters_as_bucket_view=True, **kwargs)

    param_groups = [
        {'params': modules[name].parameters(), 'lr': learning_rates[name]}
        for name in modules
    ]
    opt = mup.MuAdam(param_groups, impl=optimizer_impl, betas=(args.beta1, args.beta2))

    def compute_nll(data_iterator, steps, seq_len=args.seq_len):
        with contextlib.ExitStack() as stack:
            for ema in emas.values():
                stack.enter_context(ema.enabled())
            stack.enter_context(torch.no_grad())
            total_nll = 0.
            total_tokens = 0
            for i, X in enumerate(data_iterator):
                X = X.to(device)[:,:seq_len]
                nll = forward(x_eval=X)[1]
                total_nll += (nll.item() * X.numel())
                total_tokens += X.numel()
                if i == steps:
                    break
        return lib.ddp.reduce_mean(total_nll / total_tokens).item()

    all_val_nlls = []
    def hook(step):
        for decay in decay_to_init.values():
            decay.step(step, args.steps)

        for ema in emas.values():
            ema.step()

        if step % args.hook_freq == (args.hook_freq - 1):
            val_nll = compute_nll(val_iterator, args.val_steps)
            print(f'NLL (val, seq_len={args.seq_len}): {val_nll}')
            all_val_nlls.append(val_nll)
            if args.seq_len != 256:
                val_nll_256 = compute_nll(val_iterator, args.val_steps, seq_len=256)
                print(f'NLL (val, seq_len=256): {val_nll_256}')

            if lib.ddp.rank() == 0:
                # Save weights to checkpoint directory with _bert suffix
                if args.save_weights:
                    for name in modules:
                        with emas[name].enabled():
                            checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')
                            torch.save(modules[name].state_dict(), checkpoint_file)
                    step_file = os.path.join(checkpoint_dir, 'step')
                    with open(step_file, 'w') as f:
                        f.write(str(step))
                    # Also save a marker file to indicate this is a BERT checkpoint
                    marker_file = os.path.join(checkpoint_dir, 'tokenizer_type.txt')
                    with open(marker_file, 'w') as f:
                        f.write(f'bert\n{args.tokenizer_name}\n')
                    print(f'Saved weights to {checkpoint_dir}/ at step {step}!')

                # Save gamma plot to checkpoint directory
                t = torch.linspace(0., 1., 1024, device=device)
                gamma = modules['noise_schedule'](t)
                plt.clf()
                plt.plot(t.detach().cpu().numpy(), gamma.detach().cpu().numpy())
                plt.xlabel('t')
                plt.ylabel('gamma(t)')
                plt.title(f'Noise Schedule (step {step}, BERT)')
                gamma_file = os.path.join(checkpoint_dir, f'gamma_{step}.jpg')
                plt.savefig(gamma_file)
                plt.close()

    print('Starting train loop...')
    lib.utils.train_loop(
        forward,
        opt,
        args.steps,
        names=['nll','reconst','prior','gamma_0','gamma_1','reconst_bs'],
        hook=hook,
        print_freq=args.print_freq,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_decay=args.lr_decay,
        amp_grad_scaler=False,
        grad_accum_steps=args.grad_accum_steps,
        ddp_models=ddp_modules.values(),
        first_step=first_step,
        clip_params=[
            param
            for module in modules.values()
            for param in module.parameters()
        ],
        clip_quantile=args.clip_quantile,
    )

    final_val_nll = compute_nll(val_iterator, args.final_val_steps)
    print('Final val NLL:', final_val_nll)
    if args.seq_len != 256:
        final_val_nll_256 = compute_nll(val_iterator, args.final_val_steps, seq_len=256)
        print('Final val NLL (seq_len=256):', final_val_nll_256)

    # Save final checkpoint
    if lib.ddp.rank() == 0 and args.save_weights:
        final_step = args.steps - 1
        print(f'Saving final checkpoint at step {final_step}...')
        for name in modules:
            with emas[name].enabled():
                checkpoint_file = os.path.join(checkpoint_dir, f'{name}.pt')
                torch.save(modules[name].state_dict(), checkpoint_file)
        step_file = os.path.join(checkpoint_dir, 'step')
        with open(step_file, 'w') as f:
            f.write(str(final_step))
        # Save final model info
        info_file = os.path.join(checkpoint_dir, 'model_info.txt')
        with open(info_file, 'w') as f:
            f.write(f'Tokenizer: {args.tokenizer_name}\n')
            f.write(f'Final step: {final_step}\n')
            f.write(f'Model config: dim={args.dim}, n_blocks={args.n_blocks}, n_heads={args.n_heads}\n')
            f.write(f'Non-embedding params: {final_non_embedding:,}\n')
            f.write(f'Final val NLL: {final_val_nll}\n')
            if args.seq_len != 256:
                f.write(f'Final val NLL (seq_len=256): {final_val_nll_256}\n')
        print(f'Final checkpoint saved to {checkpoint_dir}/')

    return all_val_nlls, final_val_nll

if __name__ == '__main__':
    fire.Fire(lib.ddp.wrap_main(main))
