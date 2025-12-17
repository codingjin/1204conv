#!/usr/bin/env python3
"""
Demonstration of the ML dataset ordering logic.
This shows the exact sequence of data rows in dataset_energy.csv
"""

# Simulate the ordering logic
cases = ['case1', 'case2', 'case3']  # Example: 3 cases
kernels_per_case = 3  # Example: 3 kernels per case
powercaps = [100, 200, 300, 400, 450]  # Example: RTX 3090 with 5 power caps

print("="*80)
print("ML DATASET ORDERING DEMONSTRATION")
print("="*80)
print(f"\nConfiguration:")
print(f"  Cases: {cases}")
print(f"  Kernels per case: {kernels_per_case}")
print(f"  Power caps: {powercaps} W")
print(f"\nOrdering: case → kernel → powercap")
print("="*80)

print(f"\nDataset sequence (first 20 rows):")
print(f"{'ID':<6} {'Case':<10} {'Kernel':<10} {'PowerCap':<12} {'File Path'}")
print("-"*80)

row_id = 1
count = 0

# The CRITICAL ordering: case → kernel → powercap
for case in cases:
    for kernel_num in range(1, kernels_per_case + 1):
        for pc_idx, powercap in enumerate(powercaps, start=1):
            file_path = f"{case}/powercap{pc_idx}/output_kernel{kernel_num}.txt"

            if count < 20:  # Show first 20 rows
                print(f"{row_id:<6} {case:<10} kernel{kernel_num:<3} {powercap}W{'':<8} {file_path}")

            row_id += 1
            count += 1

print("...")
print(f"\nTotal samples: {row_id - 1}")
print(f"  = {len(cases)} cases × {kernels_per_case} kernels × {len(powercaps)} powercaps")
print(f"  = {len(cases) * kernels_per_case * len(powercaps)}")

print("\n" + "="*80)
print("KEY OBSERVATIONS:")
print("="*80)
print("1. IDs 1-5:   Same case (case1), same kernel (kernel1), different powercaps")
print("2. IDs 6-10:  Same case (case1), same kernel (kernel2), different powercaps")
print("3. IDs 16-20: Same case (case2), same kernel (kernel1), different powercaps")
print("\nThis ordering groups variations of the SAME kernel together!")
print("Perfect for ML to learn powercap → energy relationship!")
print("="*80)
