import os
import time
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
import argparse
import torch
import json

def get_gpu_sm():
    """
    Returns the compute capability of the first CUDA GPU as a string, e.g., 'sm_86'.
    Returns None if no CUDA GPU is available.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    else:
        return None

target = tvm.target.cuda(arch=get_gpu_sm())

# Convolution configurations
# Format: [N, H, W, CO, CI, KH, KW, stride, padding]
conv_configs = [
    [1, 272, 272, 64, 32, 3, 3, 1, 1],
    [1, 68, 68, 256, 128, 3, 3, 1, 1],
    [1, 34, 34, 512, 256, 3, 3, 1, 1],
    [1, 17, 17, 1024, 512, 3, 3, 1, 1],
    [1, 56, 56, 64, 64, 3, 3, 1, 1],
    [1, 28, 28, 128, 128, 3, 3, 1, 1],
    [1, 14, 14, 256, 256, 3, 3, 1, 1],
]

class Conv2DParams:
    def __init__(self, N, H, W, CO, CI, KH, KW, strides, padding):
        self.N = N
        self.H = H
        self.W = W
        self.CO = CO
        self.CI = CI
        self.KH = KH
        self.KW = KW
        self.strides = strides
        self.padding = padding
        
@auto_scheduler.register_workload
def conv2d(N, H, W, CO, CI, KH, KW, stride, padding):
    """Defines the convolution workload."""
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

def calculate_conv2d_flops(N, H, W, CO, CI, KH, KW, strides, padding):
    """
    Calculate FLOPs for Conv2D operation.

    Args:
        N: Batch size
        H: Input height
        W: Input width
        CO: Output channels
        CI: Input channels
        KH: Kernel height
        KW: Kernel width
        strides: Tuple (stride_h, stride_w)
        padding: Tuple (pad_h, pad_w)

    Returns:
        Total FLOPs
    """
    stride_h, stride_w = strides
    pad_h, pad_w = padding

    # Calculate output dimensions
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1

    # Calculate FLOPs
    # Each output element requires CI * KH * KW multiply-accumulate operations
    # Each MAC = 2 FLOPs (1 multiply + 1 add)
    flops = 2 * N * CO * OH * OW * CI * KH * KW

    return flops

def write_start_time_to_csv(log_file):
    """Writes the start time to a CSV file."""
    start_time = int(time.time())
    csv_file_path = log_file.replace('.json', '.csv')

    # write the start time to the csv file
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_file.write(f"start_time:{str(start_time)}\n")
    return start_time

def filter_tuning_results(input_json_file, output_json_file, test_mode=False):
    """Filter tuning results to keep only representative records.

    Args:
        input_json_file: Path to raw tuning results (in tuningrecords/)
        output_json_file: Path to save filtered results (in tuningresults/)
        test_mode: If True, keep only Top1 and Top2; if False, keep 25 records

    Process:
        1. Read all records
        2. Filter out "out-of-time" records (execution time >= 1e+10)
        3. Sort by execution time (ascending)
        4. Select representative records based on mode

    In test mode: keeps only Top1 (fastest) and Top2 (2nd fastest)
    In normal mode: keeps 25 representative records based on percentiles
    """
    configs = []
    out_of_time_count = 0

    # Read all records from raw tuning file
    with open(input_json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # Extract execution time from the "r" or "result" field
                if "r" in data:
                    execution_time = data["r"][0][0]
                elif "result" in data:
                    execution_time = data["result"][0][0]
                else:
                    print(f"Warning: Line {line_num} has no 'r' or 'result' field, skipping")
                    continue

                # Filter out "out-of-time" records (execution time >= 1e+10)
                if execution_time >= 1e+10:
                    out_of_time_count += 1
                    continue

                configs.append((execution_time, line.strip()))

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
                print(f"Warning: Error parsing line {line_num}: {e}")
                continue

    if not configs:
        print(f"Warning: No valid records found in {input_json_file} (filtered out {out_of_time_count} out-of-time records)")
        return

    # Sort by execution time in ascending order (fastest first)
    configs.sort(key=lambda x: x[0], reverse=False)

    total = len(configs)
    selected_indices = set()

    if test_mode:
        # Test mode: keep only Top1 (fastest) and Top2
        selected_indices.add(0)  # Top1 (fastest)
        if total >= 2:
            selected_indices.add(1)  # Top2 (index 1)
    else:
        # Normal mode: Select 25 representative records based on percentile positions
        # Top-5 (positions 0-4)
        for i in range(min(5, total)):
            selected_indices.add(i)

        # 10% position and next 4
        pos_10 = int(total * 0.10)
        for i in range(pos_10, min(pos_10 + 5, total)):
            selected_indices.add(i)

        # 25% position and next 4
        pos_25 = int(total * 0.25)
        for i in range(pos_25, min(pos_25 + 5, total)):
            selected_indices.add(i)

        # 50% position and next 4
        pos_50 = int(total * 0.50)
        for i in range(pos_50, min(pos_50 + 5, total)):
            selected_indices.add(i)

        # 75% position and next 4
        pos_75 = int(total * 0.75)
        for i in range(pos_75, min(pos_75 + 5, total)):
            selected_indices.add(i)

    # Write selected configurations to filtered output file
    selected_configs = [configs[i] for i in sorted(selected_indices)]

    with open(output_json_file, 'w') as f:
        for _, line in selected_configs:
            f.write(line + '\n')

    if out_of_time_count > 0:
        print(f"Filtered {os.path.basename(input_json_file)}: kept {len(selected_configs)} from {total} valid records ({out_of_time_count} out-of-time dropped) → {output_json_file}")
    else:
        print(f"Filtered {os.path.basename(input_json_file)}: kept {len(selected_configs)} records from {total} total → {output_json_file}")

def conv2d_tuning(specify_pz, ntrials=1000, output_dir="outputs", test_mode=False):
    """Tests the convolution workload with auto-scheduling."""

    # Override ntrials in test mode
    if test_mode:
        ntrials = 100
        print("Test mode enabled: ntrials set to 100", flush=True)

    print(f"ntrials: {ntrials}", flush=True)
    print(f"\nTesting {len(conv_configs)} convolution configurations\n")

    # Create tuningrecords directory for raw tuning data
    tuningrecords_dir = "tuningrecords"
    if not os.path.exists(tuningrecords_dir):
        os.makedirs(tuningrecords_dir)
        print(f"Created directory: {tuningrecords_dir}")

    # Create output directory for filtered results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Determine which configurations to test
    if test_mode:
        # Test mode: only first and last configurations
        print(f"Test mode: tuning only first and last configurations (case1 and case{len(conv_configs)})", flush=True)
        sizes_tmp = [conv_configs[0], conv_configs[-1]]
    elif specify_pz != -1:
        # Specific case specified
        print("testing specified case: ", specify_pz, flush=True)
        sizes_tmp = [conv_configs[int(specify_pz)]]
    else:
        # Test all cases
        print("Testing all problem sizes!", flush=True)
        sizes_tmp = conv_configs

    conv_params = {}
    for i, size in enumerate(conv_configs):
        if size not in sizes_tmp:
            continue
        N, H, W, CO, CI, KH, KW, stride, pad = size
        conv_params[i + 1] = Conv2DParams(N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad))

    for case_idx in conv_params.keys():
        conv = conv_params[case_idx]

        # Use the conv2d layer to test
        N, H, W, CO, CI, KH, KW, strides, padding = conv.N, conv.H, conv.W, conv.CO, conv.CI, conv.KH, conv.KW, conv.strides, conv.padding

        print(f"case{case_idx}, N={N}, H={H}, W={W}, CO={CO}, CI={CI}, KH={KH}, KW={KW}, strides={strides}, padding={padding}", flush=True)
        
        task = auto_scheduler.SearchTask(
            func=conv2d, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,
        )

        # Inspect the computational graph
        print("Computational DAG:", flush=True)
        print(task.compute_dag, flush=True)

        log_file = f"case{case_idx}_conv2d_N_{N}_H_{H}_W_{W}_CO_{CO}_CI_{CI}_KH_{KH}_KW_{KW}_strides_{strides}_padding_{padding}.json"
        raw_log_file = os.path.join(tuningrecords_dir, log_file)
        filtered_log_file = os.path.join(output_dir, log_file)
        
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ntrials,
            measure_callbacks=[auto_scheduler.RecordToFile(raw_log_file)],
            verbose=0, #2,
        )

        cost_model = auto_scheduler.XGBModel()
        search_policy = auto_scheduler.SketchPolicy(
            task,
            program_cost_model=cost_model,
        )

        # skip if raw_log_file already exists and lines of it is equal to ntrials
        if os.path.exists(raw_log_file):
            with open(raw_log_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == ntrials:
                    print(f"Skipping {raw_log_file} as it already has {ntrials} trials", flush=True)
                    continue
                else:
                    os.remove(raw_log_file)
                    csv_file_path = raw_log_file.replace('.json', '.csv')
                    if os.path.exists(csv_file_path):
                        os.remove(csv_file_path)

        start_time = write_start_time_to_csv(raw_log_file)

        task.tune(tune_option, search_policy)
        
        # Apply the best schedule
        try:
            sch, args = task.apply_best(raw_log_file)

            func = tvm.build(sch, args, target)

            # Check correctness
            data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
            weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
            conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
            out_np = np.maximum(conv_np, 0.0)

            dev = tvm.cuda()
            data_tvm = tvm.nd.array(data_np, device=dev)
            weight_tvm = tvm.nd.array(weight_np, device=dev)
            out_tvm = tvm.nd.empty(out_np.shape, device=dev)
            func(data_tvm, weight_tvm, out_tvm)

            # Check results
            np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)
            
            # Evaluate execution time
            evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
            print(
                f"Execution time of this operator: {np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000:.3f} ms", 
                flush=True
            )
            print(
                f"case{case_idx} conv2d for N = {N}, H = {H}, W = {W}, CO = {CO}, CI = {CI}, KH = {KH}, KW = {KW}, strides = {strides}, padding = {padding}, correctness check passed!\n",
                flush=True
            )

        except Exception as e:
            print(f"Error during tuning or execution: {e}", flush=True)
            continue

        end_time = int(time.time())
        print(f"Search time: {(end_time - start_time)/60:.2f} minutes", flush=True)

        # Filter results to keep representative kernels
        print(f"Filtering results...", flush=True)
        filter_tuning_results(raw_log_file, filtered_log_file, test_mode=test_mode)

def parse_args():
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(
        description="TVM Conv2D Auto-scheduler for various convolution configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (all problem sizes, 1000 trials)
  python conv_tuning.py

  # Test mode: quick verification (only first and last cases, 100 trials, keep only top1 and top2)
  python conv_tuning.py --test

  # Custom number of trials
  python conv_tuning.py --ntrials 2000

  # Test specific problem size (e.g., configuration index 5)
  python conv_tuning.py --specify_pz 5

  # Custom output directory
  python conv_tuning.py --output_dir my_results

  # All custom options
  python conv_tuning.py --ntrials 500 --output_dir test_results --specify_pz 3
        """
    )

    parser.add_argument(
        '--specify_pz',
        type=int,
        default=-1,
        help='Specify problem size index to test (-1 means test all, default: -1)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Enable test mode: only tune first and last configurations with 100 trials, keep only top1 and top2 in results'
    )

    parser.add_argument(
        '--ntrials',
        type=int,
        default=1000,
        help='Number of tuning trials (default: 1000, overridden to 100 in test mode)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='tuningresults',
        help='Output directory for the tuning results (default: tuningresults)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    
    # Validate arguments
    if args.ntrials <= 0:
        raise ValueError("ntrials must be a positive integer")

    # Print configuration
    print("=" * 50)
    print("TVM Conv2D Auto-scheduler Configuration")
    print("=" * 50)
    print(f"Mode: {'TEST MODE' if args.test else 'Normal'}")
    print(f"Total configurations: {len(conv_configs)}")
    print(f"Problem size index: {args.specify_pz} ({'all' if args.specify_pz == -1 else 'specific'})")
    print(f"Number of trials: {100 if args.test else args.ntrials}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)

    # Check TVM_HOME environment variable
    tvm_home = os.environ.get("TVM_HOME")
    if tvm_home:
        print(f"TVM_HOME: {tvm_home}")
    else:
        print("Warning: TVM_HOME environment variable not set")
    print("=" * 50)

    # Run tuning
    print(f"\n{'='*50}")
    print(f"Starting convolution tuning")
    print(f"{'='*50}\n")
    conv2d_tuning(args.specify_pz, args.ntrials, args.output_dir, test_mode=args.test)

    print(f"\n{'='*50}")
    print("Tuning completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*50}")
