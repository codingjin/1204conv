#!/usr/bin/env python3
"""
Complete aggregation and analysis pipeline for kernel performance and energy data.

This script performs two main tasks in sequence:
1. Parse output_kernel*.txt files and generate powercap{N}/results.csv
2. Generate all.csv from results.csv files

Output files per case:
- powercap1-5/results.csv - Raw performance and energy data per kernel
- all.csv - Combined raw data from all power caps

Usage:
  python3 generate_perfenergy.py           # Process all cases
  python3 generate_perfenergy.py case1     # Process specific case only
"""

import os
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict


# ============================================================================
# STEP 1: Parse output files and generate results.csv
# ============================================================================

def parse_output_file(file_path):
    """
    Parse a single output_kernel{K}.txt file and extract GFLOP/s, energy, and execution time.

    Returns:
        tuple: (gflops, energy_mj, exec_time_ms) or (None, None, None) if parsing fails
    """
    gflops = None
    energy_mj = None
    exec_time_ms = None

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract GFLOP/s from [Per-Iteration Performance] section
        # Pattern: "  GFLOP/s: 123.456"
        gflops_match = re.search(r'GFLOP/s:\s+([\d.]+)', content)
        if gflops_match:
            gflops = float(gflops_match.group(1))

        # Extract energy from [Per-Iteration Energy] section
        # Pattern: "  Mean energy per iteration: 12.345678000 mJ"
        energy_match = re.search(r'Mean energy per iteration:\s+([\d.]+)\s+mJ', content)
        if energy_match:
            energy_mj = float(energy_match.group(1))

        # Extract execution time from [Per-Iteration Performance] section
        # Pattern: "  Mean time per iteration: 0.123 ms"
        exec_time_match = re.search(r'Mean time per iteration:\s+([\d.]+)\s+ms', content)
        if exec_time_match:
            exec_time_ms = float(exec_time_match.group(1))

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None, None

    return gflops, energy_mj, exec_time_ms


def aggregate_powercap_directory(powercap_dir):
    """
    Aggregate all kernel results in a powercap directory.

    Args:
        powercap_dir: Path to kernel_outputs/case{M}/powercap{N}/

    Returns:
        list: List of (kernel_id, gflops, energy_mj, exec_time_ms) tuples, sorted by kernel_id
    """
    results = []

    # Find all output_kernel{K}.txt files (K starts from 1)
    output_files = sorted(powercap_dir.glob('output_kernel*.txt'))

    for output_file in output_files:
        # Extract kernel number from filename: output_kernel1.txt -> 1
        match = re.search(r'output_kernel(\d+)\.txt', output_file.name)
        if not match:
            continue

        kernel_num = int(match.group(1))
        kernel_id = kernel_num  # id = K (starts from 1)

        # Parse the file
        gflops, energy_mj, exec_time_ms = parse_output_file(output_file)

        if gflops is not None and energy_mj is not None and exec_time_ms is not None:
            results.append((kernel_id, gflops, energy_mj, exec_time_ms))
        else:
            print(f"  Warning: Could not parse {output_file}")

    # Sort by kernel_id
    results.sort(key=lambda x: x[0])

    return results


def write_results_csv(powercap_dir, results):
    """
    Write results to results.csv in the powercap directory.

    Args:
        powercap_dir: Path to kernel_outputs/case{M}/powercap{N}/
        results: List of (kernel_id, gflops, energy_mj, exec_time_ms) tuples
    """
    csv_path = powercap_dir / 'results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['id', 'perf(GFLOP/s)', 'energy(mJ)', 'EDP(ms*mJ)'])

        # Write data
        for kernel_id, gflops, energy_mj, exec_time_ms in results:
            # Round GFLOP/s to integer
            perf_int = round(gflops)
            # Keep energy with 3 decimal places
            energy_3dp = round(energy_mj, 3)
            # Calculate EDP (Energy-Delay Product) = exec_time(ms) * energy(mJ), keep 3 decimal places
            edp_3dp = round(exec_time_ms * energy_mj, 3)

            writer.writerow([kernel_id, perf_int, energy_3dp, edp_3dp])

    print(f"  ✓ Created {csv_path} ({len(results)} kernels)")


# ============================================================================
# STEP 2: Generate all.csv
# ============================================================================

def read_results_csv(filepath):
    """
    Read a results.csv file and return a dictionary mapping id -> (perf, energy, edp).

    Returns:
        dict: {id: (perf_gflops, energy_mj, edp)}
    """
    data = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel_id = int(row['id'])
                perf = float(row['perf(GFLOP/s)'])
                energy = float(row['energy(mJ)'])
                edp = float(row['EDP(ms*mJ)'])
                data[kernel_id] = (perf, energy, edp)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return {}

    return data


def generate_all_csv(case_path, powercap_data, kernel_ids):
    """
    Generate all.csv combining all powercap data.

    Args:
        case_path: Path to case directory
        powercap_data: dict mapping powercap number -> kernel data
        kernel_ids: sorted list of kernel IDs
    """
    output_file = case_path / "all.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header (simplified - no powercap prefixes)
        header = ['id']
        for pc in range(1, 6):
            header.append('perf(GFLOP/s)')
            header.append('energy(mJ)')
            header.append('EDP(ms*mJ)')
        writer.writerow(header)

        # Write data rows
        for kernel_id in kernel_ids:
            row = [kernel_id]
            for pc in range(1, 6):
                if kernel_id in powercap_data[pc]:
                    perf, energy, edp = powercap_data[pc][kernel_id]
                    row.append(f"{perf:.0f}")  # Integer
                    row.append(f"{energy:.3f}")  # 3 decimal places
                    row.append(f"{edp:.3f}")  # 3 decimal places
                else:
                    row.append("N/A")
                    row.append("N/A")
                    row.append("N/A")
            writer.writerow(row)

    print(f"  ✓ Generated: {output_file}")
    return output_file


# ============================================================================
# Main processing function
# ============================================================================

def process_case(case_dir):
    """
    Complete processing pipeline for a single case directory.

    Args:
        case_dir: Path to case directory

    Returns:
        tuple: (num_csvs_created, num_kernels_processed)
    """
    case_name = case_dir.name
    print(f"\nProcessing {case_name}/")
    print("=" * 60)

    num_csvs = 0
    num_kernels = 0

    # ========================================================================
    # STEP 1: Parse output files and generate results.csv for each power cap
    # ========================================================================
    print(f"\n[Step 1/2] Parsing output files and generating results.csv...")

    # Find all powercap directories
    powercap_dirs = sorted(case_dir.glob('powercap*'))

    if not powercap_dirs:
        print(f"  Warning: No powercap directories found in {case_name}")
        return 0, 0

    for powercap_dir in powercap_dirs:
        powercap_name = powercap_dir.name

        # Aggregate results for this powercap
        results = aggregate_powercap_directory(powercap_dir)

        if results:
            # Write CSV
            write_results_csv(powercap_dir, results)
            num_csvs += 1
            num_kernels = max(num_kernels, len(results))
        else:
            print(f"  Warning: No valid results in {case_name}/{powercap_name}")

    # ========================================================================
    # STEP 2: Generate all.csv
    # ========================================================================
    print(f"\n[Step 2/2] Generating all.csv...")

    # Read data from all 5 power caps
    powercap_data = {}
    for pc in range(1, 6):
        results_file = case_dir / f"powercap{pc}" / "results.csv"
        if not results_file.exists():
            print(f"  Warning: {results_file} not found")
            return num_csvs, num_kernels
        powercap_data[pc] = read_results_csv(results_file)

    # Check if powercap5 has data
    if not powercap_data[5]:
        print(f"  Error: No data in powercap5 for {case_name}")
        return num_csvs, num_kernels

    # Get all kernel IDs from powercap5 (reference)
    kernel_ids = sorted(powercap_data[5].keys())

    # Generate all.csv
    generate_all_csv(case_dir, powercap_data, kernel_ids)

    print(f"\n✓ Completed processing {case_name}")
    print(f"  - {num_kernels} kernels × 5 power caps")
    print(f"  - {num_csvs} results.csv files + 1 summary file")

    return num_csvs, num_kernels


def main():
    """Main function to process all cases or a specific case."""
    print("=" * 80)
    print("COMPLETE PERFORMANCE AND ENERGY DATA AGGREGATION PIPELINE")
    print("=" * 80)

    # Check if specific case_id provided as command-line argument
    target_case_id = sys.argv[1] if len(sys.argv) > 1 else None

    kernel_outputs_dir = Path('kernel_outputs')

    if not kernel_outputs_dir.exists():
        print(f"\nError: {kernel_outputs_dir} directory not found")
        print("Please run from the project root directory after running benchmarks")
        return

    # Find all case directories (or specific case if provided)
    if target_case_id:
        case_dirs = [kernel_outputs_dir / target_case_id]
        if not case_dirs[0].exists():
            print(f"\nError: {case_dirs[0]} directory not found")
            return
        print(f"\nProcessing specific case: {target_case_id}")
    else:
        case_dirs = sorted(kernel_outputs_dir.glob('case*'))
        if not case_dirs:
            print(f"\nError: No case directories found in {kernel_outputs_dir}")
            return
        print(f"\nFound {len(case_dirs)} case directories")

    # Process each case directory
    total_csvs = 0
    total_kernels = 0

    for case_dir in case_dirs:
        num_csvs, num_kernels = process_case(case_dir)
        total_csvs += num_csvs
        total_kernels = max(total_kernels, num_kernels)

    # Print final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Cases processed: {len(case_dirs)}")
    print(f"  Total results.csv files: {total_csvs}")
    print(f"  Max kernels per case: {total_kernels}")

    print(f"\nGenerated files per case:")
    print(f"  - powercap1-5/results.csv (raw performance and energy data)")
    print(f"  - all.csv (combined raw data from all power caps)")

    if case_dirs:
        print(f"\nExample location:")
        print(f"  {case_dirs[0]}/")
        print(f"    ├── powercap1/results.csv")
        print(f"    ├── powercap2/results.csv")
        print(f"    ├── powercap3/results.csv")
        print(f"    ├── powercap4/results.csv")
        print(f"    ├── powercap5/results.csv")
        print(f"    └── all.csv")
    print()


if __name__ == "__main__":
    main()
