#!/usr/bin/env python3
"""
Parse output.log file and convert to CSV, then plot different metrics.
Also upload metrics to wandb if enabled.
"""

import re
import csv
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def parse_log_file(log_file_path):
    """Parse the output.log file and extract training metrics."""
    data = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (line starting with "step")
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('step') and 'step time' in line:
            header_line_idx = i
            break
    
    if header_line_idx is None:
        raise ValueError("Could not find header line in log file")
    
    # Parse header
    header_line = lines[header_line_idx].strip()
    # Split by multiple spaces, but handle "step time" as one column
    headers = re.split(r'\s{2,}', header_line)
    # Normalize header names
    header_map = {
        'step': 'step',
        'step time': 'step_time',
        'loss': 'loss',
        'nll': 'nll',
        'reconst': 'reconst',
        'prior': 'prior',
        'gamma_0': 'gamma_0',
        'gamma_1': 'gamma_1',
        'reconst_bs': 'reconst_bs',
        'grad norm': 'grad_norm',
        'mem': 'mem'
    }
    
    # Parse data lines
    for i in range(header_line_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('NLL') or line.startswith('Saved') or line.startswith('Final') or line.startswith('Saving'):
            continue
        
        # Split by whitespace (handles both single and multiple spaces)
        parts = re.split(r'\s+', line)
        if len(parts) < 11:  # Need at least 11 columns
            continue
        
        try:
            row = {}
            row['step'] = int(parts[0])
            row['step_time'] = float(parts[1])
            row['loss'] = float(parts[2])
            row['nll'] = float(parts[3])
            row['reconst'] = float(parts[4])
            row['prior'] = float(parts[5])
            row['gamma_0'] = float(parts[6])
            row['gamma_1'] = float(parts[7])
            row['reconst_bs'] = float(parts[8])
            row['grad_norm'] = float(parts[9])
            row['mem'] = float(parts[10])
            data.append(row)
        except (ValueError, IndexError) as e:
            # Skip lines that can't be parsed
            continue
    
    return pd.DataFrame(data), headers


def save_to_csv(df, csv_file_path):
    """Save DataFrame to CSV file."""
    df.to_csv(csv_file_path, index=False)
    print(f"Saved CSV to {csv_file_path}")


def plot_metrics(df, output_dir, use_wandb=False, wandb_run=None):
    """Plot different metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('seaborn-darkgrid')
    
    # 1. Loss metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Loss Metrics', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['step'], df['loss'], label='Total Loss', linewidth=2)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['step'], df['nll'], label='NLL', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('NLL')
    axes[0, 1].set_title('Negative Log-Likelihood')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['step'], df['reconst'], label='Reconstruction Loss', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reconstruction Loss')
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['step'], df['prior'], label='Prior Loss', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Prior Loss')
    axes[1, 1].set_title('Prior Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_metrics_path = os.path.join(output_dir, 'loss_metrics.png')
    plt.savefig(loss_metrics_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss metrics plot to {loss_metrics_path}")
    if use_wandb and wandb_run:
        wandb_run.log({"plots/loss_metrics": wandb.Image(loss_metrics_path)})
    plt.close()
    
    # 2. Gamma bounds
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['step'], df['gamma_0'], label='gamma_0', linewidth=2, color='blue')
    ax.plot(df['step'], df['gamma_1'], label='gamma_1', linewidth=2, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gamma Value')
    ax.set_title('Gamma Bounds Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    gamma_bounds_path = os.path.join(output_dir, 'gamma_bounds.png')
    plt.savefig(gamma_bounds_path, dpi=300, bbox_inches='tight')
    print(f"Saved gamma bounds plot to {gamma_bounds_path}")
    if use_wandb and wandb_run:
        wandb_run.log({"plots/gamma_bounds": wandb.Image(gamma_bounds_path)})
    plt.close()
    
    # 3. Training dynamics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Dynamics', fontsize=16, fontweight='bold')
    
    axes[0, 0].plot(df['step'], df['step_time'], label='Step Time', linewidth=2, color='purple')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Step Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(df['step'], df['grad_norm'], label='Gradient Norm', linewidth=2, color='brown')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Norm')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(df['step'], df['reconst_bs'], label='Reconstruction Batch Size', linewidth=2, color='teal')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Batch Size')
    axes[1, 0].set_title('Reconstruction Batch Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(df['step'], df['mem'], label='Memory Usage', linewidth=2, color='magenta')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].set_title('GPU Memory Usage')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_dynamics_path = os.path.join(output_dir, 'training_dynamics.png')
    plt.savefig(training_dynamics_path, dpi=300, bbox_inches='tight')
    print(f"Saved training dynamics plot to {training_dynamics_path}")
    if use_wandb and wandb_run:
        wandb_run.log({"plots/training_dynamics": wandb.Image(training_dynamics_path)})
    plt.close()
    
    # 4. Combined loss comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['step'], df['loss'], label='Total Loss', linewidth=2, alpha=0.8)
    ax.plot(df['step'], df['nll'], label='NLL', linewidth=2, alpha=0.8)
    ax.plot(df['step'], df['reconst'], label='Reconstruction', linewidth=2, alpha=0.8)
    ax.plot(df['step'], df['prior'], label='Prior', linewidth=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss Value')
    ax.set_title('All Loss Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    all_losses_path = os.path.join(output_dir, 'all_losses.png')
    plt.savefig(all_losses_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined loss plot to {all_losses_path}")
    if use_wandb and wandb_run:
        wandb_run.log({"plots/all_losses": wandb.Image(all_losses_path)})
    plt.close()


def upload_to_wandb(df, log_file_path, project_name=None, run_name=None, entity=None):
    """Upload metrics to wandb."""
    if not WANDB_AVAILABLE:
        print("Error: wandb is not installed. Install with: pip install wandb")
        return None
    
    # Initialize wandb run
    config = {
        "log_file": log_file_path,
        "total_steps": len(df),
        "min_step": int(df['step'].min()),
        "max_step": int(df['step'].max()),
    }
    
    wandb_run = wandb.init(
        project=project_name or "plaid-training",
        name=run_name or os.path.basename(os.path.dirname(log_file_path)),
        entity=entity,
        config=config,
        reinit=True
    )
    
    print(f"Initialized wandb run: {wandb_run.name}")
    
    # Upload metrics step by step
    print("Uploading metrics to wandb...")
    for _, row in df.iterrows():
        wandb_run.log({
            "train/loss": row['loss'],
            "train/nll": row['nll'],
            "train/reconst": row['reconst'],
            "train/prior": row['prior'],
            "train/gamma_0": row['gamma_0'],
            "train/gamma_1": row['gamma_1'],
            "train/reconst_bs": row['reconst_bs'],
            "train/grad_norm": row['grad_norm'],
            "train/mem": row['mem'],
            "train/step_time": row['step_time'],
        }, step=int(row['step']))
    
    # Upload CSV as artifact
    csv_path = os.path.join(os.path.dirname(log_file_path), 'training_metrics.csv')
    if os.path.exists(csv_path):
        artifact = wandb.Artifact("training_metrics", type="dataset")
        artifact.add_file(csv_path)
        wandb_run.log_artifact(artifact)
        print(f"Uploaded CSV as artifact: {csv_path}")
    
    return wandb_run


def main():
    parser = argparse.ArgumentParser(description='Parse output.log and create CSV and plots')
    parser.add_argument('--log_file', type=str, 
                       default='checkpoints_bert_768_12_12_100k/output.log',
                       help='Path to output.log file')
    parser.add_argument('--output_dir', type=str, 
                       default='checkpoints_bert_768_12_12_100k',
                       help='Output directory for CSV and plots')
    parser.add_argument('--csv_name', type=str, 
                       default='training_metrics.csv',
                       help='Name of output CSV file')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Upload metrics to wandb')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='Wandb project name (default: plaid-training)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name (default: checkpoint directory name)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb entity/team name')
    
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    df, headers = parse_log_file(args.log_file)
    print(f"Parsed {len(df)} data points")
    print(f"Columns: {list(df.columns)}")
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, args.csv_name)
    save_to_csv(df, csv_path)
    
    # Upload to wandb if requested
    wandb_run = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Error: wandb is not installed. Install with: pip install wandb")
            print("Skipping wandb upload...")
        else:
            wandb_run = upload_to_wandb(
                df, 
                args.log_file,
                project_name=args.wandb_project,
                run_name=args.wandb_run_name,
                entity=args.wandb_entity
            )
    
    # Create plots
    print(f"\nCreating plots in {args.output_dir}...")
    plot_metrics(df, args.output_dir, use_wandb=args.use_wandb, wandb_run=wandb_run)
    
    # Finish wandb run
    if wandb_run:
        wandb_run.finish()
        print("Wandb run finished!")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
