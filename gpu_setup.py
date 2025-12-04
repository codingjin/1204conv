#!/usr/bin/env python3
"""
GPU Setup Script for Power Cap and Clock Configuration
Detects GPU model and applies appropriate power cap and clock settings.
"""

import subprocess
import sys
import re

# GPU Configuration Database
GPU_CONFIGS = {
    'NVIDIA GeForce RTX 3090': {
        'name': '3090',
        'power_caps': [100, 200, 300, 420, 450],  # Watts
    },
    'NVIDIA GeForce RTX 4090': {
        'name': '4090',
        'power_caps': [150, 200, 300, 400, 450],  # Watts
    },
    'Tesla V100-SXM2-16GB': {
        'name': 'V100',
        'power_caps': [100, 150, 200, 250, 300],  # Watts
    },
    'NVIDIA A30': {
        'name': 'A30',
        'power_caps': [100, 120, 140, 160, 165],  # Watts
    },
    'NVIDIA A100': {
        'name': 'A100',
        'power_caps': [100, 200, 300, 400, 450],  # Watts
    },
}

def run_command(cmd, check=True):
    """Execute shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing: {cmd}")
        print(f"Error: {e.stderr}")
        return e.stderr, e.returncode

def detect_gpu():
    """Detect GPU model using nvidia-smi."""
    cmd = "nvidia-smi --query-gpu=name --format=csv,noheader -i 0"
    output, ret = run_command(cmd)

    if ret != 0:
        print("Error: Could not detect GPU. Is nvidia-smi available?")
        sys.exit(1)

    gpu_name = output.strip()
    print(f"Detected GPU: {gpu_name}")

    # Match GPU name to configuration
    for known_gpu, config in GPU_CONFIGS.items():
        if known_gpu in gpu_name or config['name'] in gpu_name:
            print(f"Matched to configuration: {config['name']}")
            return gpu_name, config

    print(f"Error: GPU '{gpu_name}' not in supported list:")
    for gpu in GPU_CONFIGS.keys():
        print(f"  - {gpu}")
    sys.exit(1)

def count_gpus():
    """Count number of GPUs in the system."""
    cmd = "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l"
    output, ret = run_command(cmd)
    if ret != 0:
        return 1
    return int(output.strip())

def set_persistent_mode():
    """Enable persistent mode on all GPUs."""
    print("\nSetting persistent mode...")
    cmd = "sudo nvidia-smi -pm 1"
    output, ret = run_command(cmd, check=False)
    if ret != 0:
        print(f"Warning: Could not set persistent mode. May need sudo privileges.")
        print(f"Output: {output}")
    else:
        print("Persistent mode enabled.")

def disable_extra_gpus(num_gpus):
    """Disable all GPUs except device 0."""
    if num_gpus <= 1:
        print("\nOnly one GPU detected, no need to disable others.")
        return

    print(f"\nDisabling GPUs 1-{num_gpus-1} (keeping only device 0)...")
    for gpu_id in range(1, num_gpus):
        # Drain applications and set to prohibited mode
        cmd = f"sudo nvidia-smi drain -p 0 -m 1 -i {gpu_id}"
        output, ret = run_command(cmd, check=False)
        if ret != 0:
            # Try alternative method
            print(f"Warning: Could not drain GPU {gpu_id}, trying compute mode...")
            cmd = f"sudo nvidia-smi -i {gpu_id} -c PROHIBITED"
            output, ret = run_command(cmd, check=False)
            if ret != 0:
                print(f"Warning: Could not disable GPU {gpu_id}. Manual intervention may be needed.")
        else:
            print(f"GPU {gpu_id} disabled.")

def set_power_cap(power_cap_watts):
    """Set power cap on GPU 0."""
    print(f"\nSetting power cap to {power_cap_watts}W on GPU 0...")
    # Convert watts to milliwatts for nvidia-smi
    power_cap_mw = power_cap_watts * 1000
    cmd = f"sudo nvidia-smi -i 0 -pl {power_cap_watts}"
    output, ret = run_command(cmd, check=False)
    if ret != 0:
        print(f"Error: Could not set power cap to {power_cap_watts}W")
        print(f"Output: {output}")
        sys.exit(1)
    print(f"Power cap set to {power_cap_watts}W")

def verify_settings(gpu_config):
    """Verify current GPU settings."""
    print("\n" + "="*60)
    print("VERIFYING GPU SETTINGS")
    print("="*60)

    # Query power limit
    cmd = "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i 0"
    output, ret = run_command(cmd)
    if ret == 0:
        print(f"Power Limit: {output} W")

    # Query current clocks (what GPU is running at now)
    cmd = "nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader,nounits -i 0"
    output, ret = run_command(cmd)
    if ret == 0:
        clocks = output.split(',')
        if len(clocks) == 2:
            print(f"Current Clocks (Dynamic):")
            print(f"  Graphics: {clocks[0].strip()} MHz")
            print(f"  Memory: {clocks[1].strip()} MHz")

    # Query persistence mode
    cmd = "nvidia-smi --query-gpu=persistence_mode --format=csv,noheader -i 0"
    output, ret = run_command(cmd)
    if ret == 0:
        print(f"Persistence Mode: {output}")

    print("="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python gpu_setup.py <power_cap_index>")
        print("  power_cap_index: 1-5 (selects from GPU-specific power cap list)")
        print("\nOr: python gpu_setup.py --detect")
        print("  --detect: Just detect GPU and show configuration")
        sys.exit(1)

    # Detect GPU
    gpu_name, gpu_config = detect_gpu()

    if sys.argv[1] == '--detect':
        print(f"\nGPU Configuration for {gpu_config['name']}:")
        print(f"  Power Caps: {gpu_config['power_caps']} W")
        print(f"  Clock Control: Dynamic (power cap only)")
        sys.exit(0)

    # Get power cap index
    try:
        power_cap_idx = int(sys.argv[1])
        if power_cap_idx < 1 or power_cap_idx > 5:
            print("Error: power_cap_index must be between 1 and 5")
            sys.exit(1)
    except ValueError:
        print("Error: power_cap_index must be an integer")
        sys.exit(1)

    # Get power cap value from configuration
    if power_cap_idx > len(gpu_config['power_caps']):
        print(f"Error: Power cap index {power_cap_idx} not available for {gpu_config['name']}")
        print(f"Available power caps: {gpu_config['power_caps']}")
        sys.exit(1)

    power_cap = gpu_config['power_caps'][power_cap_idx - 1]

    print(f"\n{'='*60}")
    print(f"GPU SETUP FOR {gpu_config['name']}")
    print(f"{'='*60}")
    print(f"Power Cap Index: {power_cap_idx}/5")
    print(f"Target Power Cap: {power_cap}W")
    print(f"Clock Control: Dynamic (power-constrained boost)")
    print(f"{'='*60}\n")

    # Count GPUs
    num_gpus = count_gpus()
    print(f"Total GPUs in system: {num_gpus}")

    # Apply settings (order matters!)
    set_persistent_mode()
    disable_extra_gpus(num_gpus)
    set_power_cap(power_cap)

    # Verify settings
    verify_settings(gpu_config)

    print("GPU setup complete!")
    print(f"\nReady to run kernels with Power Cap {power_cap_idx} ({power_cap}W)")

if __name__ == '__main__':
    main()
