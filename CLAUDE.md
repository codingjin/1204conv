# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a TVM-based Conv2D auto-scheduler with GPU energy measurement pipeline. It:
1. Tunes Conv2D kernels using TVM auto-scheduler
2. Generates executable CUDA kernels
3. Measures energy consumption across different GPU power caps
4. Aggregates measurement results into structured CSV files

## Project Structure

### Core Scripts

**Tuning Phase:**
- `conv_tuning.py` - TVM auto-scheduler for Conv2D kernels
- `tuning_gpu_setup.sh` - GPU setup (must run before tuning)

**Kernel Generation:**
- `genkernels.py` - Generates CUDA kernels from tuning results
- `demo.cu` - Kernel wrapper template
- `main.cpp` - Energy measurement driver

**GPU Management:**
- `gpu_setup.py` - Sets GPU power caps (for measurement)
- `setup_passwordless_sudo.sh` - Configures passwordless sudo

**Post-Processing:**
- `generate_perfenergy.py` - Aggregates measurement results to CSV

**Utilities:**
- `clean.sh` - Removes all generated files

### Directory Structure

```
Source files (version controlled):
  conv_tuning.py, genkernels.py, gpu_setup.py, demo.cu, main.cpp
  generate_perfenergy.py, tuning_gpu_setup.sh, setup_passwordless_sudo.sh, clean.sh
  README.md, CLAUDE.md

Generated files (NOT version controlled):
  tuningrecords/     - Raw TVM tuning logs
  tuningresults/     - Filtered tuning results (25 or 2 kernels)
  kernels/           - Generated CUDA kernel code
  kernel_outputs/    - Energy measurement results
  build/             - Compiled binaries
  scripts/           - Generated run scripts
  run_all.sh         - Master run script
  run_layer_*.sh     - Per-case run scripts
```

## Complete Workflow

### Phase 1: TVM Tuning

**REQUIRED FIRST STEP:**
```bash
bash tuning_gpu_setup.sh
```
This sets up passwordless sudo, persistent mode, and disables extra GPUs.

**Run tuning:**
```bash
# Full tuning: 7 cases, 1000 trials each, keep 25 kernels
python conv_tuning.py

# Test mode: 2 cases (first+last), 100 trials, keep 2 kernels (Top1+Top2)
python conv_tuning.py --test
```

**Conv2D Configurations** (conv_tuning.py:26-34):
```python
conv_configs = [
    [1, 272, 272, 64, 32, 3, 3, 1, 1],    # case1
    [1, 68, 68, 256, 128, 3, 3, 1, 1],    # case2
    [1, 34, 34, 512, 256, 3, 3, 1, 1],    # case3
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],   # case4
    [1, 56, 56, 64, 64, 3, 3, 1, 1],      # case5
    [1, 28, 28, 128, 128, 3, 3, 1, 1],    # case6
    [1, 14, 14, 256, 256, 3, 3, 1, 1],    # case7
]
```
Format: `[N, H, W, CO, CI, KH, KW, stride, padding]`

**Filtering Logic** (conv_tuning.py:98-196):
1. Filters out "out-of-time" records (exec_time >= 1e10)
2. Sorts by execution time (ascending = fastest first)
3. Selects representatives:
   - **Test mode**: Top1 (fastest) + Top2 (2nd fastest)
   - **Normal mode**: 25 kernels (Top5 at 0%, 10%, 25%, 50%, 75% percentiles)

**Output Files:**
- `tuningrecords/case{N}_*.json` - Raw tuning logs (ntrials records)
- `tuningrecords/case{N}_*.csv` - Metadata with start_time
- `tuningresults/case{N}_*.json` - Filtered results (2 or 25 kernels)

**File Naming:**
```
case{N}_conv2d_N_{N}_H_{H}_W_{W}_CO_{CO}_CI_{CI}_KH_{KH}_KW_{KW}_strides_{strides}_padding_{padding}.json
```

### Phase 2: Kernel Generation

```bash
# Generate with full measurement (3 lrounds)
python genkernels.py

# Test mode (1 lround for faster validation)
python genkernels.py --test

# Custom input directory
python genkernels.py --input_dir configs
```

**What it does:**
1. Reads filtered tuning results from `tuningresults/` (default)
2. For each configuration, generates:
   - `.cuh` file - CUDA kernel header
   - `.cu` file - Kernel wrapper
   - 5 run scripts (one per power cap)
3. Filters out invalid configs (exec_time >= 1e9)
4. Generates master scripts and CMakeLists.txt

**Measurement Configuration:**
- **Normal mode**: 3 lrounds × (100 iterations × 1000 rounds) = 300,000 executions
- **Test mode**: 1 lround × (100 iterations × 1000 rounds) = 100,000 executions
- Energy measured once per lround

**Output Structure:**
```
kernels/case{N}/kernel{K}.cuh        - Kernel code
kernels/case{N}/kernel{K}.cu         - Wrapper
scripts/case{N}/run_kernel{K}_powercap{P}.sh  - Individual run scripts
run_layer_case{N}.sh                 - Run all kernels for case{N}
run_all.sh                           - Master script (all cases)
kernel_metadata.csv                  - Kernel metadata
CMakeLists.txt                       - Build configuration
```

### Phase 3: Energy Measurement

```bash
# Run all measurements (automatic passwordless sudo setup)
bash run_all.sh

# Run specific case
bash run_layer_case1.sh

# Run specific kernel with specific power cap
bash scripts/case1/run_kernel1_powercap3.sh
```

**What happens:**
1. Checks/sets up passwordless sudo
2. For each kernel × power cap:
   - Calls `gpu_setup.py {power_cap_index}` to configure GPU
   - Compiles kernel if needed
   - Runs kernel with energy measurement
   - Saves output to `kernel_outputs/case{N}/powercap{P}/output_kernel{K}.txt`

**GPU Power Caps** (gpu_setup.py:12-33):
```python
GPU_CONFIGS = {
    'NVIDIA GeForce RTX 3090': {'power_caps': [100, 200, 300, 420, 450]},
    'NVIDIA GeForce RTX 4090': {'power_caps': [150, 200, 300, 400, 450]},
    'Tesla V100-SXM2-16GB': {'power_caps': [100, 150, 200, 250, 300]},
    'NVIDIA A30': {'power_caps': [100, 120, 140, 160, 165]},
    'NVIDIA A100': {'power_caps': [100, 200, 300, 400, 450]},
}
```

### Phase 4: Post-Processing Results

```bash
# Aggregate all measurement results
python3 generate_perfenergy.py

# Or process specific case only
python3 generate_perfenergy.py case1
```

**What happens:**

**Step 1: Parse Raw Outputs**
- Reads `kernel_outputs/case{N}/powercap{1-5}/output_kernel{K}.txt`
- Extracts metrics using regex:
  - `GFLOP/s: X.XX` → Performance
  - `Mean energy per iteration: X.XX mJ` → Energy
  - `Mean time per iteration: X.XX ms` → Execution time
- Calculates **EDP (Energy-Delay Product)** = `exec_time(ms) × energy(mJ)`
- Outputs: `powercap{1-5}/results.csv` with columns:
  ```csv
  id,perf(GFLOP/s),energy(mJ),EDP(ms*mJ)
  1,1234,15.67,1.92876600
  ```

**Step 2: Generate Combined Data**
- Reads all 5 `results.csv` files (including EDP values)
- Generates `all.csv` - Combined raw data from all power caps:
  ```csv
  id,perf(GFLOP/s),energy(mJ),EDP(ms*mJ),perf(GFLOP/s),energy(mJ),EDP(ms*mJ),...
     [powercap1]                          [powercap2]
  ```
- Total columns: 1 (id) + 3 × 5 (perf/energy/EDP for each powercap) = 16 columns

**Output Structure:**
```
kernel_outputs/case1/
├── powercap1/
│   ├── output_kernel1.txt    (raw input)
│   └── results.csv           (parsed)
├── powercap2/results.csv
├── ...
├── powercap5/results.csv
└── all.csv                   (combined)
```

## Key Functions and Logic

### conv_tuning.py

**`filter_tuning_results()`** (lines 98-196):
- Reads raw tuning results
- Filters out failed tuning attempts (exec_time >= 1e10)
- Sorts by execution time (ascending)
- Selects representative kernels based on mode

**`conv2d_tuning()`** (lines 197-331):
- Main tuning loop
- Handles test mode configuration selection
- Applies filtering after tuning
- Skips if tuning already complete (checks line count)

**Test Mode Behavior** (lines 154-158, 220-223):
- Configurations: First and last (case1, case7)
- Trials: 100 (overridden from default 1000)
- Filtering: Keep Top1 and Top2 only

### genkernels.py

**Kernel Generation Loop** (lines 148-384):
- Extracts layer name from filename: `case{N}`
- Filters invalid configs: `exec_time >= 1e9`
- Generates kernel code from TVM tuning result
- Creates 5 run scripts per kernel (one per power cap)
- Stores metadata for CMakeLists.txt generation

**Grid/Block Calculation** (lines 268-285):
- Extracts from TVM schedule (SP node with 4 dimensions)
- Falls back to defaults if extraction fails: grid=1, block=256

**Test Mode** (lines 72-77):
- `num_lrounds = 1` (vs 3 in normal mode)
- Affects only measurement duration, not kernel code

### gpu_setup.py

**GPU Detection** (lines 45-66):
- Uses `nvidia-smi --query-gpu=name`
- Matches against `GPU_CONFIGS` dictionary
- Falls back to error if GPU not supported

**Power Cap Setting** (lines 108-119):
- Uses `sudo nvidia-smi -i 0 -pl {watts}`
- Applies to GPU 0 only
- Validates setting with query

**Multi-GPU Handling** (lines 87-106):
- Disables GPUs 1-N using drain or compute mode prohibited
- Ensures only GPU 0 is used for consistent measurements

## Command Reference

### conv_tuning.py

```bash
python conv_tuning.py [OPTIONS]

Options:
  --test              Test mode (2 cases, 100 trials, keep Top1+Top2)
  --ntrials N         Number of trials per case (default: 1000)
  --specify_pz N      Test specific case index (0-6)
  --output_dir DIR    Output directory (default: tuningresults)
```

### genkernels.py

```bash
python genkernels.py [OPTIONS]

Options:
  --test              Test mode (1 lround instead of 3)
  --input_dir DIR     Input directory (default: tuningresults)
```

### gpu_setup.py

```bash
python gpu_setup.py <1-5>        # Set power cap (1=lowest, 5=highest)
python gpu_setup.py --detect     # Detect GPU and show config
```

## Test Mode vs Normal Mode

| Aspect | Normal Mode | Test Mode |
|--------|-------------|-----------|
| **conv_tuning.py** | | |
| Cases | All 7 | First + last (case1, case7) |
| Trials per case | 1000 | 100 |
| Kernels kept | 25 (percentile-based) | 2 (Top1 + Top2) |
| Time estimate | 2-4 hours | 15-30 minutes |
| **genkernels.py** | | |
| Lrounds | 3 | 1 |
| Executions | 300,000 | 100,000 |
| Measurement time | Full | ~3× faster |
| **Generated kernels** | | |
| CUDA code | Identical | Identical |

## Modifying the Code

### Adding a New Conv2D Configuration

Edit `conv_configs` in `conv_tuning.py`:
```python
conv_configs = [
    [1, 272, 272, 64, 32, 3, 3, 1, 1],  # case1
    # ... existing configs ...
    [1, YOUR_H, YOUR_W, YOUR_CO, YOUR_CI, YOUR_KH, YOUR_KW, YOUR_STRIDE, YOUR_PAD],  # case8
]
```

The script automatically handles case numbering based on array position.

### Adding a New GPU

Edit `GPU_CONFIGS` in `gpu_setup.py`:
```python
GPU_CONFIGS = {
    'Full GPU Name from nvidia-smi': {
        'name': 'ShortName',
        'power_caps': [cap1, cap2, cap3, cap4, cap5],  # Watts
    },
}
```

Run `nvidia-smi --query-gpu=name --format=csv,noheader -i 0` to get exact name.

### Changing Filtering Strategy

Edit `filter_tuning_results()` in `conv_tuning.py`:
- Test mode filtering: lines 154-158
- Normal mode filtering: lines 160-183
- Adjust percentiles or count as needed

## Common Issues

### "Directory tuningresults/ not found"
- Run `conv_tuning.py` first to generate tuning results
- Or specify correct directory: `python genkernels.py --input_dir configs`

### "GPU not in supported list"
- Add GPU to `GPU_CONFIGS` in `gpu_setup.py`
- Or run `python gpu_setup.py --detect` to see detected name

### "Permission denied" when running scripts
- Make scripts executable: `chmod +x script.sh`
- Or run with bash: `bash script.sh`

### "nvidia-smi requires password"
- Run `bash tuning_gpu_setup.sh` before tuning
- Or run `bash setup_passwordless_sudo.sh` manually

## Clean Up

Remove all generated files:
```bash
bash clean.sh
```

This removes: kernels/, kernel_outputs/, build/, scripts/, metadata files, run scripts.
Does NOT remove: tuningresults/, tuningrecords/, source code.

## Requirements

- Python 3.7+
- TVM with CUDA support (set `TVM_HOME` environment variable)
- NVIDIA CUDA Toolkit
- PyTorch (for GPU detection in tuning)
- nvidia-ml-py3 (for energy measurement)
- NVIDIA GPU with compute capability >= 7.0

## Architecture Notes

- **Why two filtering modes?** Test mode (2 kernels) for quick validation, normal mode (25 kernels) for comprehensive performance analysis
- **Why separate tuningrecords/ and tuningresults/?** Keep raw data (tuningrecords/) separate from filtered data (tuningresults/) for reproducibility
- **Why disable extra GPUs?** Ensures consistent tuning results on multi-GPU systems (TVM uses all visible GPUs by default)
- **Why 5 power caps?** Provides energy/performance trade-off curve from low power to maximum performance
- **Why lrounds?** Multiple measurement rounds reduce noise in energy readings (GPU power fluctuates)
