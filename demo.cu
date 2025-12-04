
            #include <cassert>
            #include <stdlib.h>
            #include <cuda.h>
            #include <nvml.h>
            #include <cmath>
            #include <algorithm>
            #include "common.h"
            // insert headers here

void conv_kernel_energy_wrapper(int N_B, int N_C, int N_H, int N_W, int N_F, int N_R, int N_S, int PaddingH, int PaddingW,
                        int StrideH, int StrideW, int N_X, int N_Y, const float *Input,
                        const float *Kernel, float *Output, int itr, int num_lrounds) {
                        float *dev_Input_all;
                        float *dev_Kernel_all;
                        float *dev_Output_all;

                        // Fixed measurement parameters
                        const int ITERATIONS_PER_ROUND = 100;
                        const int ROUNDS_PER_LROUND = 1000;
                        const int NUM_LROUNDS = num_lrounds;  // From command-line argument (3 or 10)
                        const int executions_per_lround = ROUNDS_PER_LROUND * ITERATIONS_PER_ROUND;  // 100,000
                        const int total_executions = NUM_LROUNDS * executions_per_lround;
                        const char* mode_name = (NUM_LROUNDS == 3) ? "TEST" : "DEFAULT";

                        // Calculate sizes for single iteration
                        size_t input_size_per_iter = sizeof(float) * N_B * N_C * N_H * N_W;
                        size_t kernel_size_per_iter = sizeof(float) * N_F * N_C * N_R * N_S;
                        size_t output_size_per_iter = sizeof(float) * N_B * N_F * N_Y * N_X;
                        size_t memory_per_iter = input_size_per_iter + kernel_size_per_iter + output_size_per_iter;
                        size_t total_memory_needed = memory_per_iter * ITERATIONS_PER_ROUND;

                        // Query available GPU memory
                        size_t free_mem, total_mem;
                        CHECK(cudaMemGetInfo(&free_mem, &total_mem));
                        size_t max_usable_memory = (size_t)(free_mem * 0.9);

                        // Check if we have enough memory
                        if (total_memory_needed > max_usable_memory) {
                            fprintf(stderr, "ERROR: Required memory (%.2f GB) exceeds available GPU memory (%.2f GB)\n",
                                    (double)total_memory_needed / (1024*1024*1024),
                                    (double)max_usable_memory / (1024*1024*1024));
                            fprintf(stderr, "  Memory per iteration: %.2f MB\n",
                                    (double)memory_per_iter / (1024*1024));
                            fprintf(stderr, "  Total needed for %d iterations: %.2f GB\n",
                                    ITERATIONS_PER_ROUND, (double)total_memory_needed / (1024*1024*1024));
                            exit(1);
                        }

                        // Query and print GPU configuration
                        unsigned int power_limit_mw = 0;
                        unsigned int graphics_clock_mhz = 0;
                        unsigned int memory_clock_mhz = 0;
                        char gpu_name[NVML_DEVICE_NAME_BUFFER_SIZE];

                        nvmlReturn_t nvml_result_temp;
                        nvmlDevice_t nvml_device_temp;

                        nvml_result_temp = nvmlInit();
                        if (nvml_result_temp == NVML_SUCCESS) {
                            nvml_result_temp = nvmlDeviceGetHandleByIndex(0, &nvml_device_temp);
                            if (nvml_result_temp == NVML_SUCCESS) {
                                nvmlDeviceGetName(nvml_device_temp, gpu_name, NVML_DEVICE_NAME_BUFFER_SIZE);
                                nvmlDeviceGetPowerManagementLimit(nvml_device_temp, &power_limit_mw);
                                nvmlDeviceGetClock(nvml_device_temp, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &graphics_clock_mhz);
                                nvmlDeviceGetClock(nvml_device_temp, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &memory_clock_mhz);
                            }
                            nvmlShutdown();
                        }

                        printf("\n=== GPU Configuration ===\n");
                        printf("  GPU: %s\n", gpu_name);
                        printf("  Power Cap: %.2f W\n", (double)power_limit_mw / 1000.0);
                        printf("  Graphics Clock: %u MHz\n", graphics_clock_mhz);
                        printf("  Memory Clock: %u MHz\n", memory_clock_mhz);
                        printf("=========================\n\n");

                        printf("=== Energy Measurement Configuration ===\n");
                        printf("  Mode: %s (%d lrounds)\n", mode_name, NUM_LROUNDS);
                        printf("  Warm-up: %d iterations\n", ITERATIONS_PER_ROUND);
                        printf("  Measurement: %d lrounds × %d rounds × %d iterations = %d total executions\n",
                               NUM_LROUNDS, ROUNDS_PER_LROUND, ITERATIONS_PER_ROUND, total_executions);
                        printf("  Energy measured once per lround (%d executions)\n", executions_per_lround);
                        printf("=========================================\n\n");

                        // Allocate GPU memory
                        printf("Allocating GPU memory for %d iterations (%.2f GB total)...\n",
                               ITERATIONS_PER_ROUND, (double)total_memory_needed / (1024*1024*1024));
                        CHECK(cudaMalloc(&dev_Input_all, input_size_per_iter * ITERATIONS_PER_ROUND));
                        CHECK(cudaMalloc(&dev_Kernel_all, kernel_size_per_iter * ITERATIONS_PER_ROUND));
                        CHECK(cudaMalloc(&dev_Output_all, output_size_per_iter * ITERATIONS_PER_ROUND));

                        // Pre-load ALL iteration data to GPU (OUTSIDE energy measurement)
                        printf("Copying iteration data to GPU...\n");
                        CHECK(cudaMemcpy(dev_Input_all, Input, input_size_per_iter * ITERATIONS_PER_ROUND, cudaMemcpyHostToDevice));
                        CHECK(cudaMemcpy(dev_Kernel_all, Kernel, kernel_size_per_iter * ITERATIONS_PER_ROUND, cudaMemcpyHostToDevice));
                        CHECK(cudaMemset(dev_Output_all, 0, output_size_per_iter * ITERATIONS_PER_ROUND));
                        CHECK(cudaDeviceSynchronize());

                        // Warm-up: Execute 100 iterations before measurement
                        printf("Performing warm-up (%d iterations)...\n", ITERATIONS_PER_ROUND);
                        for (int i = 0; i < ITERATIONS_PER_ROUND; i++) {
                            // Calculate GPU memory offsets for this iteration
                            float *dev_input_ptr = dev_Input_all + i * N_B * N_C * N_H * N_W;
                            float *dev_kernel_ptr = dev_Kernel_all + i * N_F * N_C * N_R * N_S;
                            float *dev_output_ptr = dev_Output_all + i * N_B * N_F * N_Y * N_X;

                    // insert kernel call here

                        }

                        // Check for kernel errors during warm-up
                        cudaError_t warmup_err = cudaDeviceSynchronize();
                        if (warmup_err != cudaSuccess) {
                            fprintf(stderr, "\n=== KERNEL EXECUTION FAILED ===\n");
                            fprintf(stderr, "Error during warm-up: %s (code %d)\n", cudaGetErrorString(warmup_err), warmup_err);
                            fprintf(stderr, "This kernel has a TVM code generation bug causing illegal memory access.\n");
                            fprintf(stderr, "The kernel will be skipped. This does not affect other kernels.\n");
                            fprintf(stderr, "Layer params: N=%d H=%d W=%d CO=%d CI=%d KH=%d KW=%d Stride=%d Padding=%d\n",
                                    N_B, N_H, N_W, N_F, N_C, N_R, N_S, StrideH, PaddingH);
                            fprintf(stderr, "================================\n\n");
                            fflush(stderr);

                            // Cleanup and exit
                            cudaFree(dev_Input_all);
                            cudaFree(dev_Kernel_all);
                            cudaFree(dev_Output_all);
                            exit(1);
                        }
                        printf("Warm-up completed.\n");
                        printf("Starting energy measurement...\n\n");
                        fflush(stdout);

                        // Initialize NVML for energy measurement
                        nvmlReturn_t nvml_result;
                        nvmlDevice_t nvml_device;

                        nvml_result = nvmlInit();
                        if (nvml_result != NVML_SUCCESS) {
                            fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(nvml_result));
                        }

                        nvml_result = nvmlDeviceGetHandleByIndex(0, &nvml_device);
                        if (nvml_result != NVML_SUCCESS) {
                            fprintf(stderr, "Failed to get NVML device handle: %s\n", nvmlErrorString(nvml_result));
                        }

                        printf("Running %d lrounds, each with %d rounds × %d iterations...\n",
                               NUM_LROUNDS, ROUNDS_PER_LROUND, ITERATIONS_PER_ROUND);
                        printf("Total kernel executions: %d\n\n", total_executions);
                        fflush(stdout);

                        // Array to store energy per lround
                        unsigned long long energy_per_lround[NUM_LROUNDS];

                        // Run measurement lrounds
                        for (int lround = 0; lround < NUM_LROUNDS; lround++) {
                            unsigned long long energy_start, energy_end;

                            // Get energy BEFORE this lround
                            nvml_result = nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_start);
                            if (nvml_result != NVML_SUCCESS) {
                                fprintf(stderr, "Failed to get start energy: %s\n", nvmlErrorString(nvml_result));
                                energy_start = 0;
                            }

                            // Execute 20 rounds in this lround
                            for (int round = 0; round < ROUNDS_PER_LROUND; round++) {
                                // Execute 400 iterations in this round
                                for (int i = 0; i < ITERATIONS_PER_ROUND; i++) {
                                    // Calculate GPU memory offsets for this iteration
                                    float *dev_input_ptr = dev_Input_all + i * N_B * N_C * N_H * N_W;
                                    float *dev_kernel_ptr = dev_Kernel_all + i * N_F * N_C * N_R * N_S;
                                    float *dev_output_ptr = dev_Output_all + i * N_B * N_F * N_Y * N_X;

                    // insert kernel call here

                                }
                            }

                            // Synchronize to ensure all kernels in this lround completed
                            CHECK(cudaDeviceSynchronize());

                            // Get energy AFTER this lround
                            nvml_result = nvmlDeviceGetTotalEnergyConsumption(nvml_device, &energy_end);
                            if (nvml_result != NVML_SUCCESS) {
                                fprintf(stderr, "Failed to get end energy: %s\n", nvmlErrorString(nvml_result));
                                energy_end = energy_start;
                            }

                            // Store energy for this lround
                            energy_per_lround[lround] = energy_end - energy_start;
                        }

                        // Calculate statistics across lrounds
                        unsigned long long total_energy = 0;

                        for (int lr = 0; lr < NUM_LROUNDS; lr++) {
                            total_energy += energy_per_lround[lr];
                        }

                        double mean_energy_per_lround = (double)total_energy / NUM_LROUNDS;

                        // Calculate standard deviation
                        double variance = 0.0;
                        for (int lr = 0; lr < NUM_LROUNDS; lr++) {
                            double diff = (double)energy_per_lround[lr] - mean_energy_per_lround;
                            variance += diff * diff;
                        }
                        variance /= NUM_LROUNDS;
                        double std_dev = sqrt(variance);

                        // Calculate coefficient of variation
                        double cv = std_dev / mean_energy_per_lround;

                        // Calculate per-iteration energy
                        double energy_per_iteration = mean_energy_per_lround / executions_per_lround;

                        // Print statistics
                        printf("\n=== Energy Measurement Results (%d lrounds × %d rounds × %d iterations) ===\n\n",
                               NUM_LROUNDS, ROUNDS_PER_LROUND, ITERATIONS_PER_ROUND);

                        printf("[Total Energy Consumed]\n");
                        printf("  Total lrounds: %d\n", NUM_LROUNDS);
                        printf("  Total executions: %d kernels\n", total_executions);
                        printf("  Total energy: %llu mJ (%.6f J)\n\n", total_energy, (double)total_energy / 1000.0);

                        printf("[Large Round Statistics] (%d executions per lround)\n", executions_per_lround);
                        printf("  Mean energy per lround: %.6f mJ (%.9f J)\n", mean_energy_per_lround, mean_energy_per_lround / 1000.0);
                        printf("  Standard deviation: %.6f mJ (%.9f J)\n", std_dev, std_dev / 1000.0);
                        printf("  Coefficient of variation: %.6f (%.2f%%)\n\n", cv, cv * 100.0);

                        printf("[Per-Iteration Energy] (single kernel execution)\n");
                        printf("  Mean energy per iteration: %.9f mJ (%.12f J)\n\n", energy_per_iteration, energy_per_iteration / 1000.0);

                        printf("Note: NVML API returns integer mJ. Averaged values show computed precision.\n");
                        printf("==================================================================\n");
                        fflush(stdout);

                        // Cleanup NVML
                        nvmlShutdown();

                        // Copy results back to host before cleanup
                        printf("\nCopying results back to host...\n");
                        CHECK(cudaMemcpy((void*)Output, dev_Output_all, output_size_per_iter * ITERATIONS_PER_ROUND, cudaMemcpyDeviceToHost));
                        printf("Results copied successfully.\n");

                        // Free GPU memory
                        CHECK(cudaFree(dev_Input_all));
                        CHECK(cudaFree(dev_Kernel_all));
                        CHECK(cudaFree(dev_Output_all));

                    }


void conv_kernel_perf_wrapper(int N_B, int N_C, int N_H, int N_W, int N_F, int N_R, int N_S, int PaddingH, int PaddingW,
                        int StrideH, int StrideW, int N_X, int N_Y, const float *Input,
                        const float *Kernel, float *Output, int itr, int num_rounds) {
                        float *dev_Input_all;
                        float *dev_Kernel_all;
                        float *dev_Output_all;

                        // Fixed measurement parameters for performance
                        const int ITERATIONS_PER_ROUND = 100;
                        const int NUM_ROUNDS = num_rounds;  // From command-line argument
                        const int total_executions = NUM_ROUNDS * ITERATIONS_PER_ROUND;
                        const char* mode_name = (NUM_ROUNDS == 100) ? "TEST" : "DEFAULT";

                        // Calculate sizes for single iteration
                        size_t input_size_per_iter = sizeof(float) * N_B * N_C * N_H * N_W;
                        size_t kernel_size_per_iter = sizeof(float) * N_F * N_C * N_R * N_S;
                        size_t output_size_per_iter = sizeof(float) * N_B * N_F * N_Y * N_X;
                        size_t memory_per_iter = input_size_per_iter + kernel_size_per_iter + output_size_per_iter;
                        size_t total_memory_needed = memory_per_iter * ITERATIONS_PER_ROUND;

                        // Query available GPU memory
                        size_t free_mem, total_mem;
                        CHECK(cudaMemGetInfo(&free_mem, &total_mem));
                        size_t max_usable_memory = (size_t)(free_mem * 0.9);

                        // Check if we have enough memory
                        if (total_memory_needed > max_usable_memory) {
                            fprintf(stderr, "ERROR: Required memory (%.2f GB) exceeds available GPU memory (%.2f GB)\n",
                                    (double)total_memory_needed / (1024*1024*1024),
                                    (double)max_usable_memory / (1024*1024*1024));
                            fprintf(stderr, "  Memory per iteration: %.2f MB\n",
                                    (double)memory_per_iter / (1024*1024));
                            fprintf(stderr, "  Total needed for %d iterations: %.2f GB\n",
                                    ITERATIONS_PER_ROUND, (double)total_memory_needed / (1024*1024*1024));
                            exit(1);
                        }

                        // Query and print GPU configuration
                        unsigned int power_limit_mw = 0;
                        unsigned int graphics_clock_mhz = 0;
                        unsigned int memory_clock_mhz = 0;
                        char gpu_name[NVML_DEVICE_NAME_BUFFER_SIZE];

                        nvmlReturn_t nvml_result_temp;
                        nvmlDevice_t nvml_device_temp;

                        nvml_result_temp = nvmlInit();
                        if (nvml_result_temp == NVML_SUCCESS) {
                            nvml_result_temp = nvmlDeviceGetHandleByIndex(0, &nvml_device_temp);
                            if (nvml_result_temp == NVML_SUCCESS) {
                                nvmlDeviceGetName(nvml_device_temp, gpu_name, NVML_DEVICE_NAME_BUFFER_SIZE);
                                nvmlDeviceGetPowerManagementLimit(nvml_device_temp, &power_limit_mw);
                                nvmlDeviceGetClock(nvml_device_temp, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &graphics_clock_mhz);
                                nvmlDeviceGetClock(nvml_device_temp, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &memory_clock_mhz);
                            }
                            nvmlShutdown();
                        }

                        printf("\n=== GPU Configuration ===\n");
                        printf("  GPU: %s\n", gpu_name);
                        printf("  Power Cap: %.2f W\n", (double)power_limit_mw / 1000.0);
                        printf("  Graphics Clock: %u MHz\n", graphics_clock_mhz);
                        printf("  Memory Clock: %u MHz\n", memory_clock_mhz);
                        printf("=========================\n\n");

                        printf("=== Performance Measurement Configuration ===\n");
                        printf("  Mode: %s (%d rounds)\n", mode_name, NUM_ROUNDS);
                        printf("  Warm-up: %d iterations\n", ITERATIONS_PER_ROUND);
                        printf("  Measurement: %d rounds × %d iterations = %d total executions\n",
                               NUM_ROUNDS, ITERATIONS_PER_ROUND, total_executions);
                        printf("=============================================\n\n");

                        // Allocate GPU memory
                        printf("Allocating GPU memory for %d iterations (%.2f GB total)...\n",
                               ITERATIONS_PER_ROUND, (double)total_memory_needed / (1024*1024*1024));
                        CHECK(cudaMalloc(&dev_Input_all, input_size_per_iter * ITERATIONS_PER_ROUND));
                        CHECK(cudaMalloc(&dev_Kernel_all, kernel_size_per_iter * ITERATIONS_PER_ROUND));
                        CHECK(cudaMalloc(&dev_Output_all, output_size_per_iter * ITERATIONS_PER_ROUND));

                        // Pre-load ALL iteration data to GPU
                        printf("Copying iteration data to GPU...\n");
                        CHECK(cudaMemcpy(dev_Input_all, Input, input_size_per_iter * ITERATIONS_PER_ROUND, cudaMemcpyHostToDevice));
                        CHECK(cudaMemcpy(dev_Kernel_all, Kernel, kernel_size_per_iter * ITERATIONS_PER_ROUND, cudaMemcpyHostToDevice));
                        CHECK(cudaMemset(dev_Output_all, 0, output_size_per_iter * ITERATIONS_PER_ROUND));
                        CHECK(cudaDeviceSynchronize());

                        // Warm-up: Execute iterations before measurement
                        printf("Performing warm-up (%d iterations)...\n", ITERATIONS_PER_ROUND);
                        for (int i = 0; i < ITERATIONS_PER_ROUND; i++) {
                            float *dev_input_ptr = dev_Input_all + i * N_B * N_C * N_H * N_W;
                            float *dev_kernel_ptr = dev_Kernel_all + i * N_F * N_C * N_R * N_S;
                            float *dev_output_ptr = dev_Output_all + i * N_B * N_F * N_Y * N_X;

                    // insert kernel call here

                        }

                        cudaError_t warmup_err = cudaDeviceSynchronize();
                        if (warmup_err != cudaSuccess) {
                            fprintf(stderr, "\n=== KERNEL EXECUTION FAILED ===\n");
                            fprintf(stderr, "Error during warm-up: %s (code %d)\n", cudaGetErrorString(warmup_err), warmup_err);
                            fprintf(stderr, "This kernel has a bug causing illegal memory access.\n");
                            fprintf(stderr, "Layer params: N=%d H=%d W=%d CO=%d CI=%d KH=%d KW=%d Stride=%d Padding=%d\n",
                                    N_B, N_H, N_W, N_F, N_C, N_R, N_S, StrideH, PaddingH);
                            fprintf(stderr, "================================\n\n");
                            fflush(stderr);
                            cudaFree(dev_Input_all);
                            cudaFree(dev_Kernel_all);
                            cudaFree(dev_Output_all);
                            exit(1);
                        }
                        printf("Warm-up completed.\n");
                        printf("Starting performance measurement...\n\n");
                        fflush(stdout);

                        // Create CUDA events for timing
                        cudaEvent_t start, stop;
                        CHECK(cudaEventCreate(&start));
                        CHECK(cudaEventCreate(&stop));

                        // Array to store execution time per round (in milliseconds)
                        float execution_time_per_round[NUM_ROUNDS];

                        printf("Running %d rounds of %d iterations each...\n", NUM_ROUNDS, ITERATIONS_PER_ROUND);
                        printf("Total kernel executions: %d\n\n", total_executions);
                        fflush(stdout);

                        // Run measurement rounds
                        for (int round = 0; round < NUM_ROUNDS; round++) {
                            // Record start time
                            CHECK(cudaEventRecord(start));

                            // Execute iterations in this round
                            for (int i = 0; i < ITERATIONS_PER_ROUND; i++) {
                                float *dev_input_ptr = dev_Input_all + i * N_B * N_C * N_H * N_W;
                                float *dev_kernel_ptr = dev_Kernel_all + i * N_F * N_C * N_R * N_S;
                                float *dev_output_ptr = dev_Output_all + i * N_B * N_F * N_Y * N_X;

                    // insert kernel call here

                            }

                            // Record end time
                            CHECK(cudaEventRecord(stop));
                            CHECK(cudaEventSynchronize(stop));

                            // Calculate elapsed time for this round
                            float milliseconds = 0;
                            CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                            execution_time_per_round[round] = milliseconds;
                        }

                        // Calculate statistics across rounds
                        float total_time = 0.0f;
                        for (int r = 0; r < NUM_ROUNDS; r++) {
                            total_time += execution_time_per_round[r];
                        }

                        float mean_time_per_round = total_time / NUM_ROUNDS;

                        // Calculate standard deviation
                        float variance = 0.0f;
                        for (int r = 0; r < NUM_ROUNDS; r++) {
                            float diff = execution_time_per_round[r] - mean_time_per_round;
                            variance += diff * diff;
                        }
                        variance /= NUM_ROUNDS;
                        float std_dev = sqrt(variance);

                        // Calculate coefficient of variation
                        float cv = std_dev / mean_time_per_round;

                        // Calculate per-iteration time
                        float time_per_iteration = mean_time_per_round / ITERATIONS_PER_ROUND;

                        // Calculate GFLOP/s
                        // FLOPs for 2D convolution: N × F × Y × X × (2 × R × S × C)
                        // Each output element requires R × S × C multiply-add operations
                        // Each multiply-add = 2 FLOPs (1 multiply + 1 add)
                        double flops_per_conv = (double)N_B * N_F * N_Y * N_X * (2.0 * N_R * N_S * N_C);
                        double gflops = flops_per_conv / 1e9;  // Convert to GFLOPs

                        // GFLOP/s = GFLOPs / execution_time_in_seconds
                        double time_per_iteration_sec = time_per_iteration / 1000.0;  // Convert ms to seconds
                        double gflops_per_sec = gflops / time_per_iteration_sec;

                        // Print statistics
                        printf("\n=== Performance Measurement Results (%d rounds × %d iterations) ===\n\n",
                               NUM_ROUNDS, ITERATIONS_PER_ROUND);

                        printf("[Total Execution Time]\n");
                        printf("  Total rounds: %d\n", NUM_ROUNDS);
                        printf("  Total executions: %d kernels\n", total_executions);
                        printf("  Total time: %.6f ms (%.9f s)\n\n", total_time, total_time / 1000.0f);

                        printf("[Round Statistics] (%d iterations per round)\n", ITERATIONS_PER_ROUND);
                        printf("  Mean time per round: %.6f ms (%.9f s)\n", mean_time_per_round, mean_time_per_round / 1000.0f);
                        printf("  Standard deviation: %.6f ms (%.9f s)\n", std_dev, std_dev / 1000.0f);
                        printf("  Coefficient of variation: %.6f (%.2f%%)\n\n", cv, cv * 100.0f);

                        printf("[Per-Iteration Performance] (single kernel execution)\n");
                        printf("  Mean time per iteration: %.9f ms (%.12f s)\n", time_per_iteration, time_per_iteration / 1000.0f);
                        printf("  GFLOP/s: %.3f\n\n", gflops_per_sec);

                        printf("[Computational Workload]\n");
                        printf("  FLOPs per convolution: %.3e (%.3f GFLOPs)\n", flops_per_conv, gflops);
                        printf("  Convolution parameters: N=%d F=%d Y=%d X=%d R=%d S=%d C=%d\n\n",
                               N_B, N_F, N_Y, N_X, N_R, N_S, N_C);

                        printf("Note: CUDA events provide microsecond precision timing.\n");
                        printf("====================================================================\n");
                        fflush(stdout);

                        // Cleanup CUDA events
                        CHECK(cudaEventDestroy(start));
                        CHECK(cudaEventDestroy(stop));

                        // Copy results back to host
                        printf("\nCopying results back to host...\n");
                        CHECK(cudaMemcpy((void*)Output, dev_Output_all, output_size_per_iter * ITERATIONS_PER_ROUND, cudaMemcpyDeviceToHost));
                        printf("Results copied successfully.\n");

                        // Free GPU memory
                        CHECK(cudaFree(dev_Input_all));
                        CHECK(cudaFree(dev_Kernel_all));
                        CHECK(cudaFree(dev_Output_all));

                    }

