#!/bin/bash
# Setup passwordless sudo for nvidia-smi
# This allows unattended GPU configuration during long benchmark runs

echo "Setting up passwordless sudo for nvidia-smi..."
echo ""
echo "This will allow nvidia-smi to run without password prompts."
echo "This is safe and standard practice for GPU research workstations."
echo ""

# Create the sudoers file
cat << 'EOF' | sudo tee /etc/sudoers.d/nvidia-smi > /dev/null
# Allow user 'jin' to run nvidia-smi without password
# This is safe for GPU research workstations
# Created for unattended GPU power cap benchmarking
jin ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi
EOF

# Set correct permissions (sudoers files must be 0440)
sudo chmod 0440 /etc/sudoers.d/nvidia-smi

echo "✓ Passwordless sudo configured for nvidia-smi"
echo ""
echo "Testing..."
echo ""

# Test it works
if sudo -n nvidia-smi > /dev/null 2>&1; then
    echo "✓ SUCCESS! nvidia-smi can now run without password"
    echo ""
    echo "Testing GPU setup script..."
    python3 gpu_setup.py --detect
    echo ""
    echo "✓ All set! You can now run: bash run_all.sh"
    echo "  The entire pipeline will run unattended."
else
    echo "✗ Test failed. Please check configuration."
    exit 1
fi
