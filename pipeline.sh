#!/bin/bash
################################################################################
# TVM Conv2D Energy Measurement Pipeline - All-in-One Driver
#
# This script runs the complete pipeline:
#   1. GPU Setup (tuning_gpu_setup.sh)
#   2. TVM Tuning (conv_tuning.py)
#   3. Kernel Generation (genkernels.py)
#   4. Energy Measurement (run_all.sh)
#   5. Data Processing (gendata.py)
#
# Usage:
#   bash pipeline.sh                    # Run full pipeline
#   bash pipeline.sh --test             # Run in test mode
#   bash pipeline.sh --resume <stage>   # Resume from specific stage
#   bash pipeline.sh --skip-setup       # Skip GPU setup (if already done)
#
# Stages:
#   1 = GPU Setup
#   2 = TVM Tuning
#   3 = Kernel Generation
#   4 = Energy Measurement
#   5 = Data Processing
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"
START_TIME=$(date +%s)
TEST_MODE=false
SKIP_SETUP=false
RESUME_FROM=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            RESUME_FROM=2
            shift
            ;;
        --resume)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test          Run in test mode (faster, fewer kernels)"
            echo "  --skip-setup    Skip GPU setup (stage 1)"
            echo "  --resume <N>    Resume from stage N (1-5)"
            echo "  --help          Show this help message"
            echo ""
            echo "Stages:"
            echo "  1. GPU Setup"
            echo "  2. TVM Tuning"
            echo "  3. Kernel Generation"
            echo "  4. Energy Measurement"
            echo "  5. Data Processing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Functions
print_header() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_stage() {
    echo ""
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}[STAGE $1/5] $2${NC}"
    echo -e "${BLUE}────────────────────────────────────────────────────────────────────────────${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        exit 1
    fi
}

get_elapsed_time() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start pipeline
print_header "TVM Conv2D Energy Measurement Pipeline"

if [ "$TEST_MODE" = true ]; then
    print_warning "Running in TEST MODE (faster, fewer samples)"
fi

log "Pipeline started"
log "Test mode: $TEST_MODE"
log "Resume from stage: $RESUME_FROM"

# Pre-flight checks
print_info "Running pre-flight checks..."

check_command python
check_command python3
check_command nvcc
check_command nvidia-smi

if [ ! -f "conv_tuning.py" ]; then
    print_error "conv_tuning.py not found. Are you in the project directory?"
    exit 1
fi

print_success "Pre-flight checks passed"

################################################################################
# STAGE 1: GPU Setup
################################################################################

if [ $RESUME_FROM -le 1 ]; then
    print_stage 1 "GPU Setup"
    log "Stage 1: GPU Setup - Starting"

    if [ "$SKIP_SETUP" = true ]; then
        print_warning "Skipping GPU setup (--skip-setup flag)"
        log "Stage 1: GPU Setup - Skipped"
    else
        if [ ! -f "tuning_gpu_setup.sh" ]; then
            print_error "tuning_gpu_setup.sh not found"
            exit 1
        fi

        print_info "Setting up GPU for TVM tuning..."
        print_info "This may require sudo password (one time only)"

        if bash tuning_gpu_setup.sh 2>&1 | tee -a "$LOG_FILE"; then
            print_success "GPU setup completed"
            log "Stage 1: GPU Setup - Completed"
        else
            print_error "GPU setup failed"
            log "Stage 1: GPU Setup - Failed"
            exit 1
        fi
    fi

    echo ""
    print_info "Elapsed time: $(get_elapsed_time)"
fi

################################################################################
# STAGE 2: TVM Tuning
################################################################################

if [ $RESUME_FROM -le 2 ]; then
    print_stage 2 "TVM Kernel Tuning"
    log "Stage 2: TVM Tuning - Starting"

    if [ "$TEST_MODE" = true ]; then
        print_info "Running TVM tuning (TEST MODE: 2 cases, 100 trials, Top1+Top2)"
        TUNING_CMD="python conv_tuning.py --test"
    else
        print_info "Running TVM tuning (NORMAL MODE: 8 cases, 1000 trials, 25 kernels)"
        TUNING_CMD="python conv_tuning.py"
    fi

    print_info "This will take 15-30 minutes (test mode) or 2-4 hours (normal mode)"
    print_info "Progress will be logged to: $LOG_FILE"

    if $TUNING_CMD 2>&1 | tee -a "$LOG_FILE"; then
        print_success "TVM tuning completed"
        log "Stage 2: TVM Tuning - Completed"
    else
        print_error "TVM tuning failed"
        log "Stage 2: TVM Tuning - Failed"
        exit 1
    fi

    # Verify output
    if [ -d "tuningresults" ] && [ "$(ls -A tuningresults/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
        RESULT_COUNT=$(ls tuningresults/*.json 2>/dev/null | wc -l)
        print_success "Generated $RESULT_COUNT tuning result files"
    else
        print_error "No tuning results found in tuningresults/"
        exit 1
    fi

    echo ""
    print_info "Elapsed time: $(get_elapsed_time)"
fi

################################################################################
# STAGE 3: Kernel Generation
################################################################################

if [ $RESUME_FROM -le 3 ]; then
    print_stage 3 "CUDA Kernel Generation"
    log "Stage 3: Kernel Generation - Starting"

    if [ "$TEST_MODE" = true ]; then
        print_info "Generating kernels (TEST MODE: 1 lround)"
        GENKERNEL_CMD="python genkernels.py --test"
    else
        print_info "Generating kernels (NORMAL MODE: 3 lrounds)"
        GENKERNEL_CMD="python genkernels.py"
    fi

    if $GENKERNEL_CMD 2>&1 | tee -a "$LOG_FILE"; then
        print_success "Kernel generation completed"
        log "Stage 3: Kernel Generation - Completed"
    else
        print_error "Kernel generation failed"
        log "Stage 3: Kernel Generation - Failed"
        exit 1
    fi

    # Verify output
    if [ -d "kernels" ] && [ "$(find kernels -name '*.cu' 2>/dev/null | wc -l)" -gt 0 ]; then
        KERNEL_COUNT=$(find kernels -name '*.cu' 2>/dev/null | wc -l)
        print_success "Generated $KERNEL_COUNT CUDA kernels"
    else
        print_error "No kernels generated in kernels/"
        exit 1
    fi

    if [ -f "run_all.sh" ]; then
        print_success "Generated run_all.sh master script"
    else
        print_error "run_all.sh not generated"
        exit 1
    fi

    echo ""
    print_info "Elapsed time: $(get_elapsed_time)"
fi

################################################################################
# STAGE 4: Energy Measurement
################################################################################

if [ $RESUME_FROM -le 4 ]; then
    print_stage 4 "Energy Measurement"
    log "Stage 4: Energy Measurement - Starting"

    print_info "Running all kernels with energy measurement..."
    print_info "This will take several hours depending on the number of kernels"
    print_warning "Make sure the system won't sleep or lose power during measurement!"

    if [ "$TEST_MODE" = true ]; then
        print_info "Test mode: Fewer measurements (3x faster)"
    fi

    # Ask for confirmation
    echo ""
    read -p "Continue with energy measurement? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Energy measurement skipped by user"
        log "Stage 4: Energy Measurement - Skipped by user"
        print_info "To resume later, run: bash pipeline.sh --resume 4"
        exit 0
    fi

    if bash run_all.sh 2>&1 | tee -a "$LOG_FILE"; then
        print_success "Energy measurement completed"
        log "Stage 4: Energy Measurement - Completed"
    else
        print_error "Energy measurement failed"
        log "Stage 4: Energy Measurement - Failed"
        print_info "Check $LOG_FILE for details"
        print_info "To resume, run: bash pipeline.sh --resume 4"
        exit 1
    fi

    # Verify output
    if [ -d "kernel_outputs" ] && [ "$(find kernel_outputs -name 'output_kernel*.txt' 2>/dev/null | wc -l)" -gt 0 ]; then
        OUTPUT_COUNT=$(find kernel_outputs -name 'output_kernel*.txt' 2>/dev/null | wc -l)
        print_success "Generated $OUTPUT_COUNT measurement output files"
    else
        print_error "No measurement outputs found in kernel_outputs/"
        exit 1
    fi

    echo ""
    print_info "Elapsed time: $(get_elapsed_time)"
fi

################################################################################
# STAGE 5: Data Processing & ML Dataset Generation
################################################################################

if [ $RESUME_FROM -le 5 ]; then
    print_stage 5 "Data Processing & ML Dataset Generation"
    log "Stage 5: Data Processing - Starting"

    print_info "Processing measurement results and generating ML dataset..."

    if python3 gendata.py 2>&1 | tee -a "$LOG_FILE"; then
        print_success "Data processing completed"
        log "Stage 5: Data Processing - Completed"
    else
        print_error "Data processing failed"
        log "Stage 5: Data Processing - Failed"
        exit 1
    fi

    # Verify outputs
    if [ -f "dataset_energy.csv" ]; then
        SAMPLE_COUNT=$(wc -l < dataset_energy.csv)
        SAMPLE_COUNT=$((SAMPLE_COUNT - 1))  # Subtract header
        print_success "Generated dataset_energy.csv with $SAMPLE_COUNT samples"
    else
        print_error "dataset_energy.csv not generated"
        exit 1
    fi

    if [ "$(find kernel_outputs -name 'all.csv' 2>/dev/null | wc -l)" -gt 0 ]; then
        ALL_CSV_COUNT=$(find kernel_outputs -name 'all.csv' 2>/dev/null | wc -l)
        print_success "Generated $ALL_CSV_COUNT all.csv files"
    fi

    echo ""
    print_info "Elapsed time: $(get_elapsed_time)"
fi

################################################################################
# Pipeline Complete
################################################################################

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

print_header "PIPELINE COMPLETED SUCCESSFULLY"

echo -e "${GREEN}All stages completed successfully!${NC}"
echo ""
echo "Summary:"
echo "  ✓ Stage 1: GPU Setup"
echo "  ✓ Stage 2: TVM Tuning"
echo "  ✓ Stage 3: Kernel Generation"
echo "  ✓ Stage 4: Energy Measurement"
echo "  ✓ Stage 5: Data Processing"
echo ""
echo "Total elapsed time: $(get_elapsed_time)"
echo ""

print_info "Generated files:"
echo "  - tuningresults/*.json        (TVM tuning results)"
echo "  - kernels/                    (CUDA kernel code)"
echo "  - kernel_outputs/             (Raw measurement data)"
echo "  - dataset_energy.csv          (ML training dataset)"
echo ""

if [ -f "dataset_energy.csv" ]; then
    SAMPLES=$(wc -l < dataset_energy.csv)
    SAMPLES=$((SAMPLES - 1))
    echo -e "${CYAN}ML Dataset:${NC}"
    echo "  Location: dataset_energy.csv"
    echo "  Samples:  $SAMPLES"
    echo "  Format:   id,gpu,powercap(w),energy(mj)"
    echo ""
fi

print_info "Log file saved to: $LOG_FILE"
echo ""

print_header "Next Steps"
echo "1. Verify results:"
echo "   head dataset_energy.csv"
echo ""
echo "2. Start ML experiments:"
echo "   python3"
echo "   >>> import pandas as pd"
echo "   >>> df = pd.read_csv('dataset_energy.csv')"
echo "   >>> df.head()"
echo ""
echo "3. View comprehensive results:"
echo "   ls kernel_outputs/case1/all.csv"
echo ""

log "Pipeline completed successfully"
log "Total time: $(get_elapsed_time)"

print_success "All done! 🎉"
