# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a TVM-based Conv2D auto-scheduler for CUDA GPUs. It tunes convolution operations from ResNet and YOLO architectures using TVM's auto-scheduler to find optimal GPU kernel configurations.

## Running the Code

### Basic Usage

```bash
# Run with defaults (both ResNet and YOLO, all problem sizes, 1000 trials)
python conv_tuning.py

# Custom number of trials
python conv_tuning.py --ntrials 2000

# Test specific problem size (layer index)
python conv_tuning.py --specify_pz 5

# Custom output directory
python conv_tuning.py --output_dir my_results
```

### Requirements

- TVM installation (set `TVM_HOME` environment variable)
- PyTorch (for GPU capability detection)
- CUDA-capable GPU
- Python packages: `tvm`, `numpy`, `torch`

## Architecture

### Main Workflow

The script follows this pipeline:

1. **Tuning Phase** (`conv2d_tuning`): Auto-scheduler explores kernel configurations
   - Runs trials for each Conv2D layer configuration
   - Generates raw JSON logs with execution times
   - Validates correctness against reference implementation
   - Outputs to `{output_dir}/*.json` with CSV metadata

2. **Sorting Phase** (`sort_all_results`): Reduces dataset for representative samples
   - Backs up raw results to `{output_dir}.bak/`
   - Selects 25 representative configurations using percentile-based sampling
   - Overwrites `{output_dir}/*.json` with sorted, filtered results

3. **Performance Analysis** (`calculate_performance_metrics`): Converts timings to GFLOPS/s
   - Parses Conv2D parameters from filenames
   - Calculates FLOPs for each configuration
   - Outputs integer GFLOPS/s values to `performance/*.json`

### Key Components

**Problem Sizes**: Defined in `sizesResnet` (lines 30-44) and `sizesYolo` (lines 46-57)
- Format: `[N, H, W, CO, CI, KH, KW, stride, padding]`
- Most layers commented out by default (only a few active for testing)

**GPU Detection**: `get_gpu_sm()` (lines 16-25) detects compute capability
- Returns format like `sm_86` for targeted compilation
- Used globally at line 27 for `tvm.target.cuda()`

**File Naming Convention**:
- Format: `cuda_{network}_testCase_{index}_conv2d_N_{N}_H_{H}_W_{W}_CO_{CO}_CI_{CI}_KH_{KH}_KW_{KW}_strides_{strides}_padding_{padding}.json`
- Parsing regex at line 346 for extracting parameters

### Representative Sampling Strategy

The `sort_json_file` function (lines 248-311) selects 25 records from all trials:
- Top 5 fastest (positions 0-4)
- 5 records at 10th percentile
- 5 records at 25th percentile
- 5 records at 50th percentile
- 5 records at 75th percentile

This provides performance distribution insight while reducing dataset size.

## Modifying Problem Sizes

To enable/disable Conv2D layers:
1. Edit `sizesResnet` or `sizesYolo` arrays
2. Uncomment/comment layer configurations
3. Use `--specify_pz INDEX` to test individual layers during development

## Output Files

- `{output_dir}/*.json`: Sorted tuning results (25 records per layer)
- `{output_dir}.bak/*.json`: Full raw tuning results (ntrials records per layer)
- `{output_dir}/*.csv`: Metadata with `start_time` timestamp
- `performance/*.json`: GFLOPS/s values (one integer per line)
