
import os
import sys
import subprocess
import argparse

# ==========================================
# Configuration
# ==========================================
# List of r values to test
R_VALUES = [2, 6, 8, 10]

# Base directory for standard academic data (N50)
# Adjust if your server path is different
DATA_ROOT = "data/N50"

# Derived paths
RAW_TRAIN = os.path.join(DATA_ROOT, "train_raw_pyramid.pt")
# NOTE: Usually validation data is also needed. 
# We assume standard naming convention or fallback to train data if val missing (for testing).
RAW_VAL = os.path.join(DATA_ROOT, "val_raw_pyramid.pt") 
# If val raw doesn't exist, we might try to find 'test_raw_pyramid.pt' or just warn.

PRUNE_MODULE = "src.graph.prune_pyramid"
TRAIN_MODULE = "src.cli.train"

def get_pruned_path(raw_path, r):
    """
    Generate the expected filename for r-light pruned data.
    e.g. train_raw_pyramid.pt -> train_r{r}_light_pyramid.pt
    """
    head, tail = os.path.split(raw_path)
    if "raw_pyramid" in tail:
        new_tail = tail.replace("raw_pyramid", f"r{r}_light_pyramid")
    else:
        # Fallback
        new_tail = f"r{r}_{tail}"
    return os.path.join(head, new_tail)

def ensure_pruned_data(raw_path, r, env=None):
    """
    Check if pruned data exists. If not, generate it.
    Returns the path to the pruned data.
    """
    if not os.path.exists(raw_path):
        print(f"[Error] Raw data not found at: {raw_path}")
        print("Please ensure you have generated the raw pyramid data (see Makefile 'make raw').")
        sys.exit(1)

    output_path = get_pruned_path(raw_path, r)
    
    if os.path.exists(output_path):
        print(f"[Data] Found existing pruned data: {output_path}")
        return output_path
    
    print(f"[Data] Generating pruned data (r={r})...")
    print(f"       Input:  {raw_path}")
    print(f"       Output: {output_path}")
    
    # Construct command to prune
    cmd = [
        sys.executable, "-m", PRUNE_MODULE,
        "--input", raw_path,
        "--output", output_path,
        "--r", str(r),
        "--num_workers", "8" # Adjust based on server cores
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"[Data] Successfully generated: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Error] Pruning failed: {e}")
        sys.exit(1)
        
    return output_path

def run_ablation():
    # Parse CLI arguments to allow manual override (e.g., for parallel screen sessions)
    parser = argparse.ArgumentParser(description="Run ablation experiments for r.")
    parser.add_argument("--r", type=int, nargs="+", help="Specific r values to run (overrides default list).")
    # GPU selection:
    #   - Prefer --gpu for "physical" GPU id (e.g., 3 means use GPU 3 on the machine)
    #     by setting CUDA_VISIBLE_DEVICES.
    #   - Or pass --device directly (e.g., cuda:0 / cuda:3 / cpu).
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU id to use via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", type=str, default="", help="Forwarded to src.cli.train --device (e.g., cuda:0, cpu).")
    parser.add_argument("--decode_backend", type=str, default="greedy", choices=["greedy", "exact"], help="Validation decode backend forwarded to src.cli.train.")
    parser.add_argument("--exact_time_limit", type=float, default=30.0, help="Exact sparse decoder time limit in seconds.")
    parser.add_argument("--exact_length_weight", type=float, default=0.0, help="Optional Euclidean length tie-break weight for exact sparse decoding.")
    parser.add_argument("--state_mode", type=str, default="iface", choices=["iface", "matching"], help="Boundary-condition state representation forwarded to training.")
    parser.add_argument("--matching_max_used", type=int, default=4, help="Max active interfaces in matching state catalog.")
    args = parser.parse_args()

    # Use provided r values if any, otherwise use default list
    r_values_to_run = args.r if args.r else R_VALUES

    # Prepare subprocess environment / device forwarding
    env = os.environ.copy()
    device_arg = None
    if args.device:
        # Explicit device string wins
        device_arg = str(args.device)
    elif args.gpu is not None:
        # Restrict visible GPUs to a single physical id.
        # Then "cuda" / "cuda:0" in src.cli.train will map to that GPU.
        env["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
        device_arg = "cuda:0"

    if args.gpu is not None:
        print(f"[env] Using physical GPU id={int(args.gpu)} (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')})")
    if device_arg is not None:
        print(f"[env] Forwarding to {TRAIN_MODULE}: --device {device_arg}")
    
    print(f"Starting ablation for r values: {r_values_to_run}")
    print(f"Data Root: {DATA_ROOT}")
    
    # 1. Check/Prepare Data
    # If val file doesn't exist, check standard alternative 'test_raw_pyramid.pt' (often used for std datasets)
    raw_val_to_use = RAW_VAL
    if not os.path.exists(RAW_VAL):
        alt_val = os.path.join(DATA_ROOT, "test_raw_pyramid.pt")
        if os.path.exists(alt_val):
            print(f"[Info] 'val_raw_pyramid.pt' not found, using '{alt_val}' for validation.")
            raw_val_to_use = alt_val
        else:
            print(f"[Warn] No validation data found. Validation will be skipped or fail.")
            # We will let ensure_pruned_data fail if it's strictly required later, 
            # or we can set it to None and handle it.
            # For now, let's treat it as critical if missing?
            # Actually, let's just warn.
            
    for r in r_values_to_run:
        print(f"\n{'='*60}")
        print(f" ROUND: r={r}")
        print(f"{'='*60}")
        
        # A. Ensure Data Exists
        train_pt = ensure_pruned_data(RAW_TRAIN, r, env=env)
        val_pt = ensure_pruned_data(raw_val_to_use, r, env=env) if os.path.exists(raw_val_to_use) else ""

        # B. Run Training
        print(f"[Train] Starting training run for r={r}...")
        
        cmd = [
            sys.executable, "-m", TRAIN_MODULE,
            "--train_pt", train_pt,
            "--r", str(r),
            "--batch_size", "8",
            "--epochs", "20",       # As requested/implied for real experiments? Makefile says 20.
        ]

        if device_arg is not None:
            cmd.extend(["--device", device_arg])

        cmd.extend([
            "--state_mode", str(args.state_mode),
            "--matching_max_used", str(int(args.matching_max_used)),
            "--decode_backend", str(args.decode_backend),
            "--exact_time_limit", str(float(args.exact_time_limit)),
            "--exact_length_weight", str(float(args.exact_length_weight)),
        ])
        
        if val_pt:
            cmd.extend(["--val_pt", val_pt])
            
        try:
            # The training entry handles safe checkpointing (unique dir), so we just run it.
            subprocess.run(cmd, check=True, env=env)
            print(f"[Train] Finished r={r}")
        except subprocess.CalledProcessError as e:
            print(f"[Error] Training failed for r={r}: {e}")
            # Continue to next experiment? Yes.
            continue
        except KeyboardInterrupt:
            print("\n[Abort] Interrupted by user.")
            break

if __name__ == "__main__":
    run_ablation()
