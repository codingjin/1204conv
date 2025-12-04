# TVM Conv2D Auto-Scheduler with Energy Measurement

A complete pipeline for auto-tuning Conv2D kernels using TVM and measuring their energy consumption across different GPU power caps.

## Overview

This project provides:
- **TVM Auto-Scheduler**: Automated kernel optimization for Conv2D operations
- **Energy Measurement**: Per-kernel energy consumption across 5 power cap settings
- **Multi-GPU Support**: Automatic GPU configuration for consistent results
- **Unattended Execution**: Passwordless sudo setup for long-running benchmarks
- **Result Analysis**: Automated post-processing with EDP (Energy-Delay Product) calculation

## Prerequisites

- **Hardware**: NVIDIA GPU (tested on RTX 3090, RTX 4090, V100, A30, A100)
- **Software**:
  - Python 3.7+
  - TVM (with CUDA support)
  - NVIDIA CUDA Toolkit
  - nvidia-ml-py3 (for energy measurement)
  - PyTorch (for GPU detection)

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Install Python dependencies
pip install numpy tvm torch nvidia-ml-py3

# Set TVM_HOME environment variable
export TVM_HOME=/path/to/tvm
```

## Quick Start

### 1. GPU Setup (Required Before Tuning)

**Run this once before starting TVM tuning:**

```bash
bash tuning_gpu_setup.sh
```

This script:
- Sets up passwordless sudo for nvidia-smi (requires password once)
- Enables GPU persistent mode
- Disables extra GPUs on multi-GPU systems (keeps only GPU 0)

### 2. TVM Kernel Tuning

Generate optimized Conv2D kernels using TVM auto-scheduler:

```bash
# Full tuning (all 7 configurations, 1000 trials each)
python conv_tuning.py

# Test mode (first and last configurations, 100 trials, keep Top1+Top2)
python conv_tuning.py --test

# Custom configuration
python conv_tuning.py --ntrials 2000 --output_dir my_results
```

**Output**: Filtered tuning results in `tuningresults/` directory (25 kernels per case in normal mode, 2 in test mode).

### 3. Generate CUDA Kernels

Convert TVM tuning results into executable CUDA kernels:

```bash
# Generate kernels with full measurement (3 lrounds)
python genkernels.py

# Test mode (1 lround for faster validation)
python genkernels.py --test

# Custom input directory
python genkernels.py --input_dir tuningresults
```

**Output**:
- `kernels/` - CUDA kernel code
- `scripts/` - Run scripts for each kernel × power cap
- `run_all.sh` - Master script to run all measurements
- `run_layer_case{N}.sh` - Per-case measurement scripts

### 4. Run Energy Measurements

Execute kernels and measure energy consumption:

```bash
# Run all kernels × all power caps (may take hours)
bash run_all.sh

# Run specific case only
bash run_layer_case1.sh

# Run specific kernel with specific power cap
bash scripts/case1/run_kernel1_powercap3.sh
```

**Output**: Raw measurement logs in `kernel_outputs/{case_id}/powercap{1-5}/output_kernel{N}.txt`

### 5. Post-Process Results

Aggregate and analyze the measurement results:

```bash
# Process all cases
python3 generate_perfenergy.py

# Process specific case only
python3 generate_perfenergy.py case1
```

**What it does:**
1. Parses raw `output_kernel*.txt` files to extract GFLOP/s, energy (mJ), and execution time (ms)
2. Calculates EDP (Energy-Delay Product) = exec_time × energy
3. Generates `results.csv` files per power cap with columns: `id,perf(GFLOP/s),energy(mJ),EDP(ms*mJ)`
4. Combines all power caps into `all.csv` with 16 columns: `id` + (perf, energy, EDP) × 5 power caps

**Output**:
- `kernel_outputs/case{N}/powercap{1-5}/results.csv` - Parsed metrics per power cap
- `kernel_outputs/case{N}/all.csv` - Combined data from all 5 power caps

## Project Structure

```
.
├── conv_tuning.py              # TVM auto-scheduler for Conv2D
├── genkernels.py               # CUDA kernel generator
├── generate_perfenergy.py      # Post-processing and aggregation script
├── gpu_setup.py                # GPU power cap configuration (for measurement)
├── tuning_gpu_setup.sh         # GPU setup for TVM tuning (run first!)
├── setup_passwordless_sudo.sh  # Standalone passwordless sudo setup
├── clean.sh                    # Remove all generated files
├── demo.cu                     # Kernel wrapper template
├── main.cpp                    # Energy measurement driver
├── README.md                   # This file
├── CLAUDE.md                   # Context for Claude Code (optional)
│
├── tuningresults/              # Filtered tuning results (version control: no)
├── tuningrecords/              # Raw tuning logs (version control: no)
├── kernels/                    # Generated CUDA kernels (version control: no)
├── kernel_outputs/             # Measurement results (version control: no)
├── build/                      # Compiled binaries (version control: no)
└── scripts/                    # Generated run scripts (version control: no)
```

## Detailed Workflow

### Phase 1: TVM Tuning

**Before tuning (required):**
```bash
bash tuning_gpu_setup.sh
```

**Run tuning:**
```bash
python conv_tuning.py [--test] [--ntrials N] [--output_dir DIR]
```

**What happens:**
1. Tunes Conv2D kernels using TVM auto-scheduler
2. Filters results to keep representative kernels:
   - **Normal mode**: 25 kernels (Top5 at 0%, 10%, 25%, 50%, 75% percentiles)
   - **Test mode**: 2 kernels (Top1 and Top2)
3. Saves filtered results to `tuningresults/case{N}_*.json`

**Conv2D Configurations:**
- 7 configurations covering typical CNN layers
- Format: `[N, H, W, CO, CI, KH, KW, stride, padding]`
- Examples: 272×272×64, 68×68×256, 34×34×512, etc.

### Phase 2: Kernel Generation

```bash
python genkernels.py [--test] [--input_dir DIR]
```

**What happens:**
1. Reads tuning results from `tuningresults/` (default)
2. Generates CUDA kernel code for each configuration
3. Creates run scripts with 5 power cap variants each
4. Builds directory structure for measurements

**Measurement Configuration:**
- **Normal mode**: 3 lrounds × 100,000 executions = 300,000 total per kernel
- **Test mode**: 1 lround × 100,000 executions = 100,000 total per kernel

### Phase 3: Energy Measurement

```bash
bash run_all.sh
```

**What happens:**
1. Sets up passwordless sudo (if not already done)
2. For each kernel and each power cap (1-5):
   - Configures GPU with `gpu_setup.py`
   - Compiles kernel (if needed)
   - Runs kernel with energy measurement
   - Saves results to `kernel_outputs/`

**Power Cap Settings:**
- GPU-specific power caps (5 levels per GPU type)
- Examples: RTX 3090: [100W, 200W, 300W, 420W, 450W]

### Phase 4: Post-Processing

```bash
# Process all cases
python3 generate_perfenergy.py

# Or process specific case
python3 generate_perfenergy.py case1
```

**What happens:**
1. **Step 1: Parse Raw Outputs**
   - Reads `kernel_outputs/case{N}/powercap{1-5}/output_kernel{K}.txt`
   - Extracts GFLOP/s, energy (mJ), and execution time (ms) using regex
   - Calculates EDP (Energy-Delay Product) = exec_time × energy
   - Generates `results.csv` per power cap with columns: `id,perf(GFLOP/s),energy(mJ),EDP(ms*mJ)`

2. **Step 2: Generate Combined Data**
   - Reads all 5 `results.csv` files
   - Combines into `all.csv` with 16 columns: `id` + (perf, energy, EDP) × 5 power caps
   - Each row contains complete data for one kernel across all power cap settings

**Output:**
- `kernel_outputs/case{N}/powercap{1-5}/results.csv` - Parsed metrics per power cap
- `kernel_outputs/case{N}/all.csv` - Combined data from all 5 power caps

## Scripts Reference

### Setup Scripts

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `tuning_gpu_setup.sh` | Setup GPU for TVM tuning | **Before `conv_tuning.py`** |
| `setup_passwordless_sudo.sh` | Standalone sudo setup | Optional (included in `tuning_gpu_setup.sh`) |
| `gpu_setup.py` | Set GPU power cap for measurement | Automatically called by run scripts |

### Main Scripts

| Script | Purpose | Options |
|--------|---------|---------|
| `conv_tuning.py` | TVM auto-scheduler | `--test`, `--ntrials N`, `--output_dir DIR` |
| `genkernels.py` | Generate CUDA kernels | `--test`, `--input_dir DIR` |
| `generate_perfenergy.py` | Post-process results | `[case_id]` (optional, process specific case) |
| `clean.sh` | Remove generated files | None (interactive) |

### Test Mode Comparison

| Aspect | Normal Mode | Test Mode |
|--------|-------------|-----------|
| **conv_tuning.py** | | |
| Configurations | All 7 cases | First + last (case1, case7) |
| Trials per config | 1000 | 100 |
| Kernels kept | 25 (percentile-based) | 2 (Top1 + Top2) |
| **genkernels.py** | | |
| Lrounds | 3 | 1 |
| Total executions | 300,000 | 100,000 |
| **Time Estimate** | | |
| Tuning | ~2-4 hours | ~15-30 minutes |
| Measurement | Hours (depends on kernels) | ~3× faster |

## Output Files

### After Tuning

```
tuningrecords/
└── case{N}_conv2d_*.json          # Raw tuning logs (.json + .csv)

tuningresults/
└── case{N}_conv2d_*.json          # Filtered results (25 or 2 kernels)
```

### After Kernel Generation

```
kernels/case{N}/
├── kernel1.cuh                    # CUDA kernel header
├── kernel1.cu                     # Kernel wrapper
└── ...

scripts/case{N}/
├── run_kernel1_powercap1.sh       # Individual run scripts
├── run_kernel1_powercap2.sh       # (one per kernel × power cap)
└── ...

run_layer_case{N}.sh               # Run all kernels for case{N}
run_all.sh                         # Master script
kernel_metadata.csv                # Kernel metadata
CMakeLists.txt                     # CMake build config
```

### After Measurements

```
kernel_outputs/case{N}/
├── powercap1/
│   ├── output_kernel1.txt         # Raw measurement output
│   ├── output_kernel2.txt
│   ├── ...
│   └── results.csv                # Parsed metrics (id,perf,energy,EDP)
├── powercap2/results.csv
├── powercap3/results.csv
├── powercap4/results.csv
├── powercap5/results.csv
└── all.csv                        # Combined data from all power caps
```

**Raw output file format** (example):
```
[Per-Iteration Performance]
  GFLOP/s: 1234.56

[Per-Iteration Energy]
  Mean energy per iteration: 12.34 mJ
```

## GPU Configuration

The project auto-detects and configures the following GPUs:

| GPU Model | Power Caps (W) |
|-----------|----------------|
| RTX 3090 | 100, 200, 300, 420, 450 |
| RTX 4090 | 150, 200, 300, 400, 450 |
| V100 | 100, 150, 200, 250, 300 |
| A30 | 100, 120, 140, 160, 165 |
| A100 | 100, 200, 300, 400, 450 |

**To add a new GPU:**
Edit `gpu_setup.py` and add entry to `GPU_CONFIGS` dictionary.

## Troubleshooting

### Issue: "nvidia-smi requires sudo password"

**Solution:**
```bash
bash tuning_gpu_setup.sh
# OR
bash setup_passwordless_sudo.sh
```

### Issue: "TVM_HOME environment variable not set"

**Solution:**
```bash
export TVM_HOME=/path/to/tvm
# Add to ~/.bashrc for persistence
echo 'export TVM_HOME=/path/to/tvm' >> ~/.bashrc
```

### Issue: "No JSON files found in tuningresults"

**Solution:**
1. Run `conv_tuning.py` first to generate tuning results
2. Check `--output_dir` matches `--input_dir` in `genkernels.py`

### Issue: Multi-GPU system using wrong GPU

**Solution:**
```bash
bash tuning_gpu_setup.sh  # Disables extra GPUs
```

To re-enable all GPUs:
```bash
sudo nvidia-smi -c DEFAULT
sudo nvidia-smi -pm 0
```

## Clean Up

Remove all generated files:

```bash
bash clean.sh
```

This removes:
- `kernels/`, `kernel_outputs/`, `build/`, `scripts/`
- `kernel_metadata.csv`, `CMakeLists.txt`
- `run_all.sh`, `run_layer_*.sh`

**Does NOT remove:**
- Tuning results (`tuningresults/`, `tuningrecords/`)
- Source code

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{tvm-conv2d-energy,
  title={TVM Conv2D Auto-Scheduler with Energy Measurement},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yourrepo}
}
```

## License

[Your chosen license]

## Contact

[Your contact information]

## Acknowledgments

- [Apache TVM](https://tvm.apache.org/) for the auto-scheduling framework
- NVIDIA Management Library (NVML) for energy measurement APIs
