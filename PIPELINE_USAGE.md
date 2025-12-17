# Pipeline Script - Quick Reference

## All-in-One Driver for TVM Conv2D Energy Measurement

The `pipeline.sh` script automates the entire workflow from GPU setup to ML dataset generation.

---

## Basic Usage

### Run Full Pipeline
```bash
bash pipeline.sh
```
Runs all 5 stages:
1. GPU Setup
2. TVM Tuning (8 cases, 1000 trials, 25 kernels)
3. Kernel Generation (3 lrounds)
4. Energy Measurement (all kernels × all power caps)
5. Data Processing (combine tuning results + generate dataset_energy.csv)

**Time**: 4-8 hours (depending on GPU and number of kernels)

---

### Test Mode (Faster)
```bash
bash pipeline.sh --test
```
Same stages but with reduced workload:
- Only 2 cases (case1, case8)
- Only 100 trials per case
- Only 2 kernels per case (Top1, Top2)
- Only 1 lround for measurements

**Time**: 1-2 hours

---

## Advanced Usage

### Skip GPU Setup
```bash
bash pipeline.sh --skip-setup
```
Use this if you've already run GPU setup before.

### Resume from Specific Stage
```bash
bash pipeline.sh --resume 3
```
Resume from stage 3 (Kernel Generation) if earlier stages are already complete.

**Stages**:
- 1 = GPU Setup
- 2 = TVM Tuning
- 3 = Kernel Generation
- 4 = Energy Measurement
- 5 = Data Processing

### Help
```bash
bash pipeline.sh --help
```

---

## Features

### Progress Tracking
- Colored output (green = success, red = error, yellow = warning)
- Stage-by-stage progress indicators
- Elapsed time display
- Clear success/failure messages

### Error Handling
- Automatic error detection at each stage
- Stops execution on error
- Provides clear error messages
- Suggests recovery commands

### Logging
- All output saved to timestamped log file: `pipeline_YYYYMMDD_HHMMSS.log`
- Includes timestamps for each event
- Useful for debugging and auditing

### Verification
- Checks for required commands (python, nvcc, nvidia-smi)
- Verifies file existence before proceeding
- Counts generated files after each stage
- Validates output structure

---

## Stage Details

### Stage 1: GPU Setup
**What it does**:
- Sets up passwordless sudo for nvidia-smi
- Enables GPU persistent mode
- Disables extra GPUs (keeps only GPU 0)

**Duration**: 1-2 minutes

**Skip if**: You've already run `tuning_gpu_setup.sh`

---

### Stage 2: TVM Tuning
**What it does**:
- Runs TVM auto-scheduler on Conv2D configurations
- Tunes kernels for optimal performance
- Filters and saves top kernels

**Output**: `tuningresults/*.json`

**Duration**:
- Test mode: 15-30 minutes
- Normal mode: 2-4 hours

---

### Stage 3: Kernel Generation
**What it does**:
- Converts TVM tuning results to CUDA code
- Generates run scripts for each kernel × power cap
- Creates CMakeLists.txt for building

**Output**:
- `kernels/` - CUDA kernel code
- `scripts/` - Run scripts
- `run_all.sh` - Master script

**Duration**: 1-5 minutes

---

### Stage 4: Energy Measurement
**What it does**:
- Compiles and runs all kernels
- Measures energy at different power caps
- Saves raw measurement data

**Output**: `kernel_outputs/case*/powercap*/output_kernel*.txt`

**Duration**:
- Test mode: 30-60 minutes
- Normal mode: 2-4 hours

**Note**: Requires user confirmation before starting (long-running process)

---

### Stage 5: Data Processing
**What it does**:
- Combines all tuning results into single file
- Parses raw measurement outputs
- Generates per-powercap CSV files
- Generates combined all.csv files
- Creates ML training dataset

**Output**:
- `allkernels.json.{GPU}` (combined tuning results)
- `kernel_outputs/case*/powercap*/results.csv`
- `kernel_outputs/case*/all.csv`
- `dataset_energy.csv` (ML dataset)

**Duration**: 1-2 minutes

---

## Example Workflows

### First Time Setup (Full Pipeline)
```bash
# Run everything from scratch
bash pipeline.sh

# Estimated time: 6-10 hours
```

### Quick Test Run
```bash
# Test the pipeline with minimal data
bash pipeline.sh --test

# Estimated time: 1-2 hours
```

### Resume After Interruption
```bash
# If pipeline stopped at stage 4
bash pipeline.sh --resume 4
```

### Re-run Data Processing Only
```bash
# If you only need to regenerate CSV files
bash pipeline.sh --resume 5
```

### Multiple Tuning Experiments
```bash
# Initial run
bash pipeline.sh

# Later: re-generate kernels and measure again
bash pipeline.sh --resume 3 --skip-setup
```

---

## Output Files

After successful completion, you'll have:

```
tuningresults/
├── case1_conv2d_*.json
├── case2_conv2d_*.json
└── ...

kernels/
├── case1/
│   ├── kernel1.cuh
│   ├── kernel1.cu
│   └── ...
└── ...

kernel_outputs/
├── case1/
│   ├── powercap1/
│   │   ├── output_kernel1.txt
│   │   └── results.csv
│   ├── powercap2/
│   │   └── results.csv
│   ├── ...
│   └── all.csv
└── ...

allkernels.json.{GPU}      # Combined tuning results
dataset_energy.csv         # ML training dataset
pipeline_*.log             # Execution log
```

---

## Troubleshooting

### "Command not found: python"
**Solution**: Install Python 3.7+
```bash
which python3
python3 --version
```

### "nvidia-smi not found"
**Solution**: Install NVIDIA drivers and CUDA toolkit

### "TVM_HOME not set"
**Solution**: Set TVM_HOME environment variable
```bash
export TVM_HOME=/path/to/tvm
```

### "Permission denied"
**Solution**: Make script executable
```bash
chmod +x pipeline.sh
```

### "Stage 4 takes too long"
**Solution**: Use test mode or run fewer cases
```bash
bash pipeline.sh --test
```

### Pipeline interrupted
**Solution**: Resume from last completed stage
```bash
# Check log file to see which stage completed
tail pipeline_*.log

# Resume (e.g., from stage 3)
bash pipeline.sh --resume 3
```

---

## Best Practices

1. **Start with test mode** to verify everything works
   ```bash
   bash pipeline.sh --test
   ```

2. **Use screen/tmux** for long-running processes
   ```bash
   screen -S pipeline
   bash pipeline.sh
   # Press Ctrl+A, D to detach
   # screen -r pipeline to reattach
   ```

3. **Monitor progress** in another terminal
   ```bash
   tail -f pipeline_*.log
   ```

4. **Check intermediate results** after each stage
   ```bash
   ls tuningresults/
   ls kernels/
   ls kernel_outputs/
   ```

5. **Save logs** for future reference
   ```bash
   cp pipeline_*.log saved_logs/
   ```

---

## Time Estimates

### Test Mode (`--test`)
| Stage | Time |
|-------|------|
| 1. GPU Setup | 1-2 min |
| 2. TVM Tuning | 15-30 min |
| 3. Kernel Gen | 1 min |
| 4. Measurement | 30-60 min |
| 5. Processing | 1 min |
| **Total** | **1-2 hours** |

### Normal Mode (default)
| Stage | Time |
|-------|------|
| 1. GPU Setup | 1-2 min |
| 2. TVM Tuning | 2-4 hours |
| 3. Kernel Gen | 2-5 min |
| 4. Measurement | 2-4 hours |
| 5. Processing | 1-2 min |
| **Total** | **6-10 hours** |

*Times vary based on GPU, number of cases, and system load*

---

## See Also

- `README.md` - Full project documentation
- `CLAUDE.md` - Detailed implementation guide
- `ML_DATASET_DOCUMENTATION.md` - ML dataset reference
