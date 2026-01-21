import contextlib
import fire
import mup
import numpy as np
import lib.datasets
import lib.models
import lib.utils
import os
import time
import torch
import torch.nn.functional as F
import transformers
import tqdm
from torch import nn, optim, autograd

def main(**args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = lib.utils.AttributeDict(args)
    args.setdefault('seq_len', 128)
    args.setdefault('vocab_size', None)  # Will be auto-detected from tokenizer or checkpoint
    args.setdefault('weights_path', None)
    args.setdefault('checkpoint_dir', None)  # If None, will infer from weights_path
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('initial_noise_scale', 1.0)
    args.setdefault('n_samples', 1024)
    args.setdefault('batch_size', 32)  # Batch size for generation to avoid OOM
    args.setdefault('sampling_timesteps', 4096)
    args.setdefault('score_temp', 0.9)
    args.setdefault('output_scale', 1.)
    args.setdefault('owt2_tokenizer', None)  # None means auto-detect from checkpoint
    args.setdefault('tokenizer_name', 'google-bert/bert-base-uncased')
    args.setdefault('ddim_sampler', False)
    args.setdefault('guidance_weight', 2.)
    
    # Determine checkpoint directory
    if args.checkpoint_dir is None:
        if args.weights_path is None:
            raise ValueError("Either checkpoint_dir or weights_path must be provided")
        args.checkpoint_dir = args.weights_path
    # Convert to absolute path
    if not os.path.isabs(args.checkpoint_dir):
        args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    
    # Auto-detect tokenizer type from checkpoint if not specified
    if args.owt2_tokenizer is None:
        tokenizer_type_file = os.path.join(args.checkpoint_dir, 'tokenizer_type.txt')
        if os.path.exists(tokenizer_type_file):
            with open(tokenizer_type_file, 'r') as f:
                lines = f.readlines()
                tokenizer_type = lines[0].strip() if len(lines) > 0 else 'owt2'
                if len(lines) > 1:
                    args.tokenizer_name = lines[1].strip()
            args.owt2_tokenizer = (tokenizer_type != 'bert')
            print(f'Auto-detected tokenizer type: {tokenizer_type}, tokenizer_name: {args.tokenizer_name}')
        else:
            # Default to BERT if no tokenizer_type.txt found
            args.owt2_tokenizer = False
            print(f'No tokenizer_type.txt found, defaulting to BERT tokenizer: {args.tokenizer_name}')

    lib.utils.print_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device('cuda')

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    # Initialize tokenizer early to get vocab_size
    if args.owt2_tokenizer:
        tokenizer = lib.datasets.openwebtext2_tokenizer()
        print('Using OWT2 tokenizer')
        if args.vocab_size is None:
            args.vocab_size = 32768  # OWT2 default vocab_size
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=True,
        )
        # Ensure pad_token is set for BERT
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            elif tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
        print(f'Using BERT tokenizer: {args.tokenizer_name}')
        # Get vocab_size from tokenizer
        if args.vocab_size is None:
            args.vocab_size = len(tokenizer.get_vocab())
            print(f'Auto-detected vocab_size from tokenizer: {args.vocab_size}')
    
    # Try to get vocab_size from checkpoint config.json if available
    if args.weights_path is not None:
        config_file = os.path.join(args.checkpoint_dir, 'config.json')
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if 'vocab_size' in config:
                        checkpoint_vocab_size = config['vocab_size']
                        if args.vocab_size is not None and args.vocab_size != checkpoint_vocab_size:
                            print(f'Warning: vocab_size mismatch. Tokenizer: {args.vocab_size}, Checkpoint: {checkpoint_vocab_size}. Using checkpoint vocab_size.')
                        args.vocab_size = checkpoint_vocab_size
                        print(f'Using vocab_size from checkpoint config: {args.vocab_size}')
            except Exception as e:
                print(f'Warning: Failed to read vocab_size from config.json: {e}')
    
    # Final fallback: try to infer from embedding_matrix checkpoint
    if args.vocab_size is None and args.weights_path is not None:
        embedding_file = os.path.join(args.weights_path, 'embedding_matrix.pt')
        if os.path.exists(embedding_file):
            try:
                embedding_state = torch.load(embedding_file, map_location='cpu')
                if 'matrix' in embedding_state:
                    args.vocab_size = embedding_state['matrix'].shape[0]
                    print(f'Auto-detected vocab_size from embedding_matrix checkpoint: {args.vocab_size}')
            except Exception as e:
                print(f'Warning: Failed to infer vocab_size from embedding_matrix: {e}')
    
    # Final fallback to default
    if args.vocab_size is None:
        args.vocab_size = 32768
        print(f'Warning: Using default vocab_size: {args.vocab_size}')

    def log1mexp(x):
        # Computes log(1-exp(-|x|))
        x = -x.abs()
        return torch.where(
            x > -0.693,
            torch.log(-torch.expm1(x)),
            torch.log1p(-torch.exp(x))
        )

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(args.vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, args.vocab_size).float()
        }
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.cuda()

    print(f'Loading weights from {args.weights_path}')
    for name, module in modules.items():
        module.load_state_dict(torch.load(
            os.path.join(args.weights_path, f'{name}.pt'),
            map_location=torch.device('cuda')
        ))

    for key in modules:
        print(key+':')
        lib.utils.print_model(modules[key])


    def generate_samples(guidance_tokens, seq_len=args.seq_len, n_samples=None):
        """
        Sampling (implements Appendix A.4 eqn 33 in VDM). Needs float64 to work.
        guidance_tokens: [(token, weight, position, complement), ...]
            token: vocab index of token
            weight: guidance weight
            position: sequence index, or 'any', or 'all'
            complement: if True, do guidance on log(1-p(y|x))
        n_samples: number of samples to generate (if None, uses args.n_samples)
        """
        if n_samples is None:
            n_samples = args.n_samples
        with torch.no_grad():
            embedding_matrix = modules['embedding_matrix']()

            gamma_0, gamma_1 = modules['gamma_bounds']()
            alpha_0 = torch.sigmoid(-gamma_0).sqrt()
            sigma_0 = torch.sigmoid(gamma_0).sqrt()

            z = torch.randn((n_samples, seq_len, args.embed_dim), device='cuda') * args.initial_noise_scale
            x_selfcond = torch.zeros_like(z).float()
            for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps))):
                t = t[None].cuda()
                s = t - 1. / args.sampling_timesteps
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                if len(guidance_tokens) > 0:
                    with torch.enable_grad():
                        z.requires_grad = True
                        logits, x_reconst = modules['model'](
                            z=z.to(torch.float32, copy=True),
                            gamma=gamma_t.float(),
                            embedding_matrix=embedding_matrix,
                            bias_scale=1.,
                            x_selfcond=x_selfcond
                        )

                        logprobs = F.log_softmax(logits.float(), dim=2)
                        logprobs_any = logprobs.logsumexp(dim=1)-float(seq_len)

                        sum_logp = 0.
                        for token, weight, position, complement in guidance_tokens:
                            if position == 'any':
                                logp = logprobs_any[:, token]
                            elif position == 'all':
                                logp = logprobs[:, :, token]
                            else:
                                logp = logprobs[:, position, token]
                            if complement:
                                logp = log1mexp(logp)
                            sum_logp += weight * logp.sum()

                        guidance_grad = autograd.grad(sum_logp, [z])[0]
                        z.requires_grad = False
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                    x_reconst += guidance_grad.double() * sigma_squared_t / alpha_squared_t.sqrt()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                else:
                    _, x_reconst = modules['model'](
                        z=z.to(torch.float32, copy=True),
                        gamma=gamma_t.float(),
                        embedding_matrix=embedding_matrix,
                        bias_scale=1.,
                        x_selfcond=x_selfcond
                    )
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                if t > 0:
                    if args.ddim_sampler:
                        z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
                    else:
                        c = -torch.expm1(gamma_s - gamma_t)
                        z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                        z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                        z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)

            logits, _ = modules['model'](
                z=z.float(),
                gamma=gamma_t.float(),
                embedding_matrix=embedding_matrix,
                bias_scale=1.,
                x_selfcond=x_selfcond
            )
            x_samples = logits.argmax(dim=-1)

            return x_samples

    def decode_samples(x_samples):
        """Decode samples and return list of text strings"""
        decoded_texts = []
        if args.owt2_tokenizer:
            for x in x_samples:
                x = tokenizer.decode(x.tolist(), skip_special_tokens=False)
                decoded_texts.append(x)
        else:
            # Use BERT tokenizer - skip special tokens like [CLS], [SEP], [PAD] for cleaner output
            for x in x_samples:
                token_ids = x.tolist()
                decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
                decoded_texts.append(decoded_text)
        return decoded_texts
    
    def print_samples(x_samples):
        """Print samples (for backward compatibility)"""
        decoded_texts = decode_samples(x_samples)
        for text in decoded_texts:
            print(text.replace("\n", "↵"))
        return decoded_texts
    
    print(f'Generating {args.n_samples} unconditional samples with seq_len={args.seq_len} (batch_size={args.batch_size})...')
    
    # Generate samples in batches to avoid OOM
    all_samples = []
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, args.n_samples)
        batch_size_actual = batch_end - batch_start
        
        print(f'Generating batch {batch_idx + 1}/{n_batches} (samples {batch_start}-{batch_end-1})...')
        batch_samples = generate_samples([], seq_len=args.seq_len, n_samples=batch_size_actual)
        all_samples.append(batch_samples.cpu())
        
        # Clear GPU cache after each batch
        torch.cuda.empty_cache()
    
    # Concatenate all batches
    x_samples = torch.cat(all_samples, dim=0)
    print(f'Generated {x_samples.shape[0]} samples total')
    
    # Save samples as token IDs (for perplexity calculation)
    token_ids_file = os.path.join(args.checkpoint_dir, 'samples_token_ids.pt')
    try:
        torch.save(x_samples, token_ids_file)
        print(f'Saved token IDs to {token_ids_file}')
    except Exception as e:
        print(f'Warning: Failed to save token IDs to file: {e}')
    
    # Also save as text (one sample per line)
    text_file = os.path.join(args.checkpoint_dir, 'samples.txt')
    try:
        decoded_texts = decode_samples(x_samples)
        with open(text_file, 'w', encoding='utf-8') as f:
            for text in decoded_texts:
                f.write(text + '\n')
        print(f'Saved {len(decoded_texts)} samples as text to {text_file}')
    except Exception as e:
        print(f'Warning: Failed to save samples as text to file: {e}')
    
    # Print first few samples for verification
    print('\nFirst 5 samples:')
    decoded_texts = decode_samples(x_samples[:5])
    for i, text in enumerate(decoded_texts, 1):
        print(f'\nSample {i}:')
        print(text.replace("\n", "↵"))


if __name__ == '__main__':
    fire.Fire(main)