#include <string.h>
#include <stdlib.h>     /* abs */
#include <math.h>
#include "common.h"

int main(int argc, char *argv[]) {
    if(argc < 11){
        printf("Missing parameters ");
        printf("%d provided, needs 10\n", argc-1);
        return 1;
    }
    int N_B = atoi(argv[1]);
    int N_H = atoi(argv[2]);
    int N_W = atoi(argv[3]);
    int N_F = atoi(argv[4]);
    int N_C = atoi(argv[5]);
    int N_R = atoi(argv[6]);
    int N_S = atoi(argv[7]);
    int strides = atoi(argv[8]);
    int padding = atoi(argv[9]);
    int num_lrounds = atoi(argv[10]);  // Number of large measurement rounds
    int N_X = ((N_W - N_S + 2 * padding) / strides + 1); /*output x*/
    int N_Y = ((N_H - N_R + 2 * padding) / strides + 1); /*output y*/
    float *Input;
    float *Kernel;
    float *Output;

    int itr = 100;  // Fixed 100 iterations for measurement
    generate_input_tensor(N_B, N_C, N_H, N_W, &Input, itr);
    generate_kernel(N_F, N_C, N_R, N_S, &Kernel, itr);
    size_t output_size = (size_t)N_B * N_F * N_Y * N_X * itr;
    Output = (TYPE *) malloc(sizeof(TYPE) * output_size);

    // Performance measurement: use 100 rounds for test mode (3 lrounds), 1000 rounds for default (10 lrounds)
    int num_perf_rounds = (num_lrounds == 3) ? 100 : 1000;

    printf("\n========================================\n");
    printf("  STARTING PERFORMANCE MEASUREMENT\n");
    printf("========================================\n");
    conv_kernel_perf_wrapper(N_B, N_C, N_H, N_W, N_F, N_R, N_S, padding, padding,
                        strides, strides, N_X, N_Y, Input, Kernel, Output, itr, num_perf_rounds);

    printf("\n========================================\n");
    printf("  STARTING ENERGY MEASUREMENT\n");
    printf("========================================\n");
    conv_kernel_energy_wrapper(N_B, N_C, N_H, N_W, N_F, N_R, N_S, padding, padding,
                        strides, strides, N_X, N_Y, Input, Kernel, Output, itr, num_lrounds);
    return 0;

}
