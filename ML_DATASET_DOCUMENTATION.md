# ML Dataset Generation - Documentation

## Overview

The `gendata.py` script now generates a machine learning training dataset (`dataset_energy.csv`) for predicting CUDA kernel energy consumption.

## Output File: `dataset_energy.csv`

### Format
```csv
id,gpu,powercap(w),energy(mj)
1,RTX3090,100,16.123
2,RTX3090,200,18.456
3,RTX3090,300,20.789
...
```

### Columns

1. **id** (integer): Sequential identifier starting from 1
2. **gpu** (string): GPU model name (RTX3090, RTX4090, V100, A30, A100)
3. **powercap(w)** (integer): Power cap setting in watts
4. **energy(mj)** (float, 3 decimals): Energy consumption in millijoules

## Critical: Data Ordering

The dataset follows a **strict hierarchical ordering** designed for ML:

```
case1 → case2 → case3 → ... → caseN
  ↓
  kernel1 → kernel2 → kernel3 → ... → kernelM
    ↓
    powercap1 → powercap2 → powercap3 → ... → powercapP
```

### Example Sequence (RTX 3090 with 5 power caps):

```
ID   Case    Kernel   PowerCap   Source File
1    case1   kernel1  100W       case1/powercap1/output_kernel1.txt
2    case1   kernel1  200W       case1/powercap2/output_kernel1.txt
3    case1   kernel1  300W       case1/powercap3/output_kernel1.txt
4    case1   kernel1  400W       case1/powercap4/output_kernel1.txt
5    case1   kernel1  450W       case1/powercap5/output_kernel1.txt
6    case1   kernel2  100W       case1/powercap1/output_kernel2.txt
7    case1   kernel2  200W       case1/powercap2/output_kernel2.txt
...
```

### Why This Ordering?

**Grouping by kernel configuration** allows ML models to:
- Learn the relationship between power cap and energy for identical kernels
- Recognize that consecutive rows (IDs 1-5) are the same kernel at different power levels
- Exploit sequential patterns in the data

## GPU-Specific Behavior

The dataset automatically adapts to the detected GPU:

### RTX 3090
```python
GPU: RTX3090
Power caps: [100, 200, 300, 400, 450] W
Samples per kernel: 5
```

### RTX 4090
```python
GPU: RTX4090
Power caps: [150, 200, 300, 400, 450] W
Samples per kernel: 5
```

### NVIDIA A30
```python
GPU: A30
Power caps: [100, 130, 165] W
Samples per kernel: 3  # Note: Fewer samples!
```

### Tesla V100
```python
GPU: V100
Power caps: [100, 150, 200, 250, 300] W
Samples per kernel: 5
```

### NVIDIA A100
```python
GPU: A100
Power caps: [100, 200, 250, 300, 400] W
Samples per kernel: 5
```

## Dataset Size Calculation

```
Total Samples = (Number of Cases) × (Kernels per Case) × (Power Caps per GPU)
```

**Example**: 8 cases, 25 kernels per case, RTX 3090 (5 power caps)
```
Total = 8 × 25 × 5 = 1,000 samples
```

**Example**: 8 cases, 25 kernels per case, A30 (3 power caps)
```
Total = 8 × 25 × 3 = 600 samples
```

## Usage

### Generate Everything (Standard Pipeline)
```bash
python3 gendata.py
```

This will:
1. Parse all output files
2. Generate `results.csv` for each powercap
3. Generate `all.csv` for each case
4. **Generate `dataset_energy.csv` for ML training**

### Generate for Specific Case Only
```bash
python3 gendata.py case1
```

This processes only `case1` but still generates the complete ML dataset.

## Data Quality & Error Handling

### Automatic Validation
- **Missing files**: Skipped with warning, count tracked
- **Parse failures**: Skipped with warning, count tracked
- **Invalid values**: Filtered out (energy must be numeric)

### Output Summary
```
ML DATASET GENERATION COMPLETE
================================================================================
Output file: dataset_energy.csv
Total samples: 1000
Skipped files: 3
GPU: RTX3090
Power cap range: 100W - 450W

Dataset structure:
  - 8 cases
  - ~25 kernels per case
  - 5 power caps per kernel
  - Expected samples per case: ~125
================================================================================
```

## ML Model Considerations

### Features (Input)
- **gpu**: Categorical feature (5 classes)
- **powercap(w)**: Continuous feature (range varies by GPU)
- **kernel configuration**: Not included in this file, join with `kernel_metadata.csv`

### Target (Output)
- **energy(mj)**: Continuous regression target

### Recommended Preprocessing
1. **One-hot encode** `gpu` column
2. **Normalize** `powercap(w)` to [0, 1] range
3. **Log-transform** `energy(mj)` if distribution is skewed
4. **Add kernel features** by joining with `kernel_metadata.csv`:
   - N, H, W, CO, CI, KH, KW (convolution parameters)
   - Grid/block dimensions
   - GPU architecture (sm_XX)

### Example Join
```python
import pandas as pd

# Load ML dataset
ml_data = pd.read_csv('dataset_energy.csv')

# Load kernel metadata
metadata = pd.read_csv('kernel_metadata.csv')

# Extract case and kernel info from file paths
# (requires parsing case{M}/powercap{N}/output_kernel{K}.txt pattern)
# Then join on case_id + kernel_idx
```

## File Location

- **Generated at**: Project root directory
- **Filename**: `dataset_energy.csv`
- **Overwritten**: Each run overwrites the previous file

## Verification

To verify correct ordering, run the test script:
```bash
python3 test_ml_dataset_ordering.py
```

This shows the first 20 rows with their source files.

## Advanced: Multi-GPU Datasets

To create a multi-GPU dataset:

1. **Run on each GPU separately**:
   ```bash
   # On RTX 3090
   python3 gendata.py
   mv dataset_energy.csv dataset_energy_rtx3090.csv

   # On A30
   python3 gendata.py
   mv dataset_energy.csv dataset_energy_a30.csv
   ```

2. **Combine datasets**:
   ```python
   import pandas as pd

   df_3090 = pd.read_csv('dataset_energy_rtx3090.csv')
   df_a30 = pd.read_csv('dataset_energy_a30.csv')

   # Reset IDs to be globally unique
   df_a30['id'] = df_a30['id'] + df_3090['id'].max()

   # Combine
   combined = pd.concat([df_3090, df_a30], ignore_index=True)
   combined.to_csv('dataset_energy_multi_gpu.csv', index=False)
   ```

## Troubleshooting

### "No case directories found"
- Ensure you run from the project root
- Check that `kernel_outputs/` exists
- Verify that measurement scripts have been run

### "Could not parse energy"
- Check that output files contain the expected format
- Look for `Mean energy per iteration: X.XX mJ` in output files
- Verify that kernels completed successfully

### "Skipped files: 100+"
- Check that all power cap measurements completed
- Verify that `run_all.sh` ran successfully
- Look for failed kernel executions in output files

### Different GPU detected than expected
- The script auto-detects the current GPU
- If running on different hardware, dataset will reflect that GPU
- Use `python3 gpu_setup.py --detect` to verify GPU detection

## Future Extensions

Potential enhancements:
1. **Add more features**: Include kernel parameters (N, H, W, etc.)
2. **Add performance data**: Include GFLOP/s and EDP
3. **Add temporal data**: Include execution time
4. **Normalization options**: Built-in feature scaling
5. **Train/test split**: Automatic split generation
6. **Data augmentation**: Synthetic samples via interpolation
