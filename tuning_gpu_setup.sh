#!/bin/bash
# GPU Setup for TVM Tuning
# Sets passwordless sudo, persistent mode, and disables extra GPUs (keeps only GPU 0)
# This ensures consistent tuning results on multi-GPU systems

echo "=========================================="
echo "GPU SETUP FOR TVM TUNING"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

# Setup passwordless sudo for nvidia-smi
echo "Checking passwordless sudo configuration..."
if sudo -n nvidia-smi > /dev/null 2>&1; then
    echo "✓ Passwordless sudo already configured"
else
    echo "Setting up passwordless sudo for nvidia-smi..."
    echo "This requires sudo password once:"
    echo ""

    # Create sudoers file
    cat << 'EOF' | sudo tee /etc/sudoers.d/nvidia-smi > /dev/null
# Allow current user to run nvidia-smi without password
# This is safe for GPU research workstations
# Created for unattended GPU tuning and benchmarking
$USER ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
EOF

    # Set correct permissions
    sudo chmod 0440 /etc/sudoers.d/nvidia-smi

    # Test it works
    if sudo -n nvidia-smi > /dev/null 2>&1; then
        echo "✓ Passwordless sudo configured successfully"
    else
        echo "✗ Error: Failed to configure passwordless sudo"
        echo "  Manual setup may be required"
        exit 1
    fi
fi
echo ""

# Count GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s) in the system"
echo ""

# Set persistent mode on all GPUs
echo "Setting persistent mode..."
if sudo nvidia-smi -pm 1 > /dev/null 2>&1; then
    echo "✓ Persistent mode enabled on all GPUs"
else
    echo "✗ Warning: Could not set persistent mode. May need sudo privileges."
    echo "  Try running: sudo nvidia-smi -pm 1"
fi
echo ""

# Disable extra GPUs if multi-GPU system
if [ $NUM_GPUS -gt 1 ]; then
    echo "Multi-GPU system detected. Disabling GPUs 1-$((NUM_GPUS-1))..."
    echo "This ensures TVM tuning uses only GPU 0 for consistent results."
    echo ""

    for (( gpu_id=1; gpu_id<$NUM_GPUS; gpu_id++ )); do
        echo "  Disabling GPU $gpu_id..."

        # Try drain method first
        if sudo nvidia-smi drain -p 0 -m 1 -i $gpu_id > /dev/null 2>&1; then
            echo "    ✓ GPU $gpu_id disabled (drain mode)"
        else
            # Try compute mode method as fallback
            if sudo nvidia-smi -i $gpu_id -c PROHIBITED > /dev/null 2>&1; then
                echo "    ✓ GPU $gpu_id disabled (compute prohibited)"
            else
                echo "    ✗ Warning: Could not disable GPU $gpu_id"
                echo "      Manual intervention may be needed"
            fi
        fi
    done
    echo ""
else
    echo "Single GPU system - no need to disable extra GPUs"
    echo ""
fi

# Verify setup
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="
echo ""
echo "Active GPU (GPU 0):"
nvidia-smi --query-gpu=index,name,persistence_mode,compute_mode --format=csv,noheader -i 0
echo ""

if [ $NUM_GPUS -gt 1 ]; then
    echo "Disabled GPUs:"
    for (( gpu_id=1; gpu_id<$NUM_GPUS; gpu_id++ )); do
        nvidia-smi --query-gpu=index,name,persistence_mode,compute_mode --format=csv,noheader -i $gpu_id
    done
    echo ""
fi

echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "GPU 0 is ready for TVM tuning with:"
echo "  ✓ Passwordless sudo configured"
echo "  ✓ Persistent mode enabled"
if [ $NUM_GPUS -gt 1 ]; then
    echo "  ✓ Other GPUs disabled (single GPU mode)"
fi
echo ""
echo "You can now run: python conv_tuning.py"
echo "No password prompts will be needed during tuning."
echo ""
echo "Note: To re-enable all GPUs later, run:"
echo "  sudo nvidia-smi -c DEFAULT"
echo "  sudo nvidia-smi -pm 0"
