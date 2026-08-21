#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INPUT_MATRIX_SIZE 28
#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER_SIZE 128
#define OUTPUT_LAYER_SIZE 10
#define BLOCK_SIZE 16
#define MU 0.1307f
#define SIGMA 0.3081f
#define CHECK_CUDA_ERROR(val)             \
    {                                     \
        check((val), __FILE__, __LINE__); \
    }

inline void check(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Fatal Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// 1st host layer
float *h_input_layer_1, *h_weight_layer_12, *h_bias_layer_12;
// 2nd host layer
float *h_output_layer_2, *h_weight_layer_23, *h_bias_layer_23;

// 1st device layer
float *d_input_layer_1, *d_input_layer_2, *d_weight_layer_12, *d_bias_layer_12;
// 2nd device layer
float *d_output_layer_2, *d_weight_layer_23, *d_bias_layer_23;
int PADDED_BATCH_SIZE;

// __global__ void mac_kernel(float *A, float *B, float *C, float *bias, int M, int K, int N)
// {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     if (row < M && col < N)
//     {
//         float sum = 0.0f;
//         for (int i = 0; i < K; i++)
//         {
//             sum += A[row * K + i] * B[i * N + col];
//         }
//         C[row * N + col] = sum + bias[row];
//     }
// }

__global__ void add_bias(float *A, float *bias, int size, int columns)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        A[idx] += bias[idx % columns];
    }
}

__global__ void ReLU(float *A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        if (A[idx] < 0)
        {
            A[idx] = 0;
        }
    }
}

__global__ void normalise_kernel(float *A, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        A[idx] = (A[idx] - MU) / SIGMA;
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

inline void allocate_memory()
{
    // allocate memory for input layer,hidden layer, output layer, and weight layer on both host and device
    h_input_layer_1 = (float *)malloc(PADDED_BATCH_SIZE * INPUT_LAYER_SIZE * (sizeof(float)));
    h_weight_layer_12 = (float *)malloc(INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
    h_bias_layer_12 = (float *)malloc(HIDDEN_LAYER_SIZE * sizeof(float));

    h_output_layer_2 = (float *)malloc(PADDED_BATCH_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));
    h_weight_layer_23 = (float *)malloc(HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));
    h_bias_layer_23 = (float *)malloc(OUTPUT_LAYER_SIZE * sizeof(float));

    CHECK_CUDA_ERROR(cudaMalloc(&d_input_layer_1, PADDED_BATCH_SIZE * INPUT_LAYER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_input_layer_2, PADDED_BATCH_SIZE * HIDDEN_LAYER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight_layer_12, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias_layer_12, HIDDEN_LAYER_SIZE * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMalloc(&d_output_layer_2, PADDED_BATCH_SIZE * OUTPUT_LAYER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weight_layer_23, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bias_layer_23, OUTPUT_LAYER_SIZE * sizeof(float)));
}

void read_file(char *file, float *array, size_t size)
{
    FILE *opened_file = fopen(file, "rb");
    if (opened_file == NULL)
    {
        printf("Error opening file for reading\n");
        exit(1);
    }
    size_t read_el = fread(array, sizeof(float), size, opened_file);
    if (read_el != size)
    {
        printf("Error reading file: expected %zu elements, got %zu\n", size, read_el);
        exit(1);
    }
    fclose(opened_file);
}
inline void initialize_weights()
{
    read_file((char *)"binary_files/MLP.0.weight.bin", h_weight_layer_12, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE);
    read_file((char *)"binary_files/MLP.0.bias.bin", h_bias_layer_12, HIDDEN_LAYER_SIZE);
    read_file((char *)"binary_files/MLP.2.weight.bin", h_weight_layer_23, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);
    read_file((char *)"binary_files/MLP.2.bias.bin", h_bias_layer_23, OUTPUT_LAYER_SIZE);
}

inline void move_data_to_device()
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_input_layer_1, h_input_layer_1, PADDED_BATCH_SIZE * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight_layer_12, h_weight_layer_12, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_weight_layer_23, h_weight_layer_23, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias_layer_12, h_bias_layer_12, HIDDEN_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias_layer_23, h_bias_layer_23, OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice));
}

inline void forward_pass(cublasHandle_t handle)
{
    float alpha = 1.0f, beta = 0.0f;
    // 1st layer
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_LAYER_SIZE, PADDED_BATCH_SIZE, INPUT_LAYER_SIZE, &alpha, d_weight_layer_12, HIDDEN_LAYER_SIZE, d_input_layer_1, INPUT_LAYER_SIZE, &beta, d_input_layer_2, HIDDEN_LAYER_SIZE);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    // relu
    dim3 threadperBlockReLU(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocksperGridRelu((HIDDEN_LAYER_SIZE * PADDED_BATCH_SIZE + threadperBlockReLU.x - 1) / threadperBlockReLU.x);
    ReLU<<<blocksperGridRelu, threadperBlockReLU>>>(d_input_layer_2, HIDDEN_LAYER_SIZE * PADDED_BATCH_SIZE);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    // add bias
    dim3 threadperBlockBias1(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocksperGridBias1((HIDDEN_LAYER_SIZE * PADDED_BATCH_SIZE + threadperBlockBias1.x - 1) / threadperBlockBias1.x);
    add_bias<<<blocksperGridBias1, threadperBlockBias1>>>(d_input_layer_2, d_bias_layer_12, HIDDEN_LAYER_SIZE * PADDED_BATCH_SIZE, HIDDEN_LAYER_SIZE);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    // 2nd layer
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_LAYER_SIZE, PADDED_BATCH_SIZE, HIDDEN_LAYER_SIZE, &alpha, d_weight_layer_23, OUTPUT_LAYER_SIZE, d_input_layer_2, HIDDEN_LAYER_SIZE, &beta, d_output_layer_2, OUTPUT_LAYER_SIZE);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());

    // add bias 2
    dim3 threadperBlockBias2(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocksperGridBias2((PADDED_BATCH_SIZE * OUTPUT_LAYER_SIZE + threadperBlockBias2.x - 1) / threadperBlockBias2.x);
    add_bias<<<blocksperGridBias2, threadperBlockBias2>>>(d_output_layer_2, d_bias_layer_23, PADDED_BATCH_SIZE * OUTPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

inline void free_memory()
{
    free(h_input_layer_1);
    free(h_weight_layer_12);
    free(h_bias_layer_12);
    free(h_output_layer_2);
    free(h_weight_layer_23);
    free(h_bias_layer_23);
    CHECK_CUDA_ERROR(cudaFree(d_input_layer_1));
    CHECK_CUDA_ERROR(cudaFree(d_input_layer_2));
    CHECK_CUDA_ERROR(cudaFree(d_weight_layer_12));
    CHECK_CUDA_ERROR(cudaFree(d_output_layer_2));
    CHECK_CUDA_ERROR(cudaFree(d_weight_layer_23));
    CHECK_CUDA_ERROR(cudaFree(d_bias_layer_12));
    CHECK_CUDA_ERROR(cudaFree(d_bias_layer_23));
}

int main()
{
    int BATCH_SIZE;
    printf("Enter the batch size: ");
    scanf("%d", &BATCH_SIZE);
    printf("\n");
    PADDED_BATCH_SIZE = (BATCH_SIZE + 15) / BLOCK_SIZE * BLOCK_SIZE;
    allocate_memory();
    cublasHandle_t handle;
    cublasCreate(&handle);
    // give input
    read_file((char *)"binary_files/input.bin", h_input_layer_1, BATCH_SIZE * INPUT_LAYER_SIZE);
    // initialize weights and biases
    initialize_weights();
    double start_inference_time = get_time();
    move_data_to_device();
    cudaMemset(d_input_layer_1 + (BATCH_SIZE * INPUT_LAYER_SIZE), 0.0, ((PADDED_BATCH_SIZE - BATCH_SIZE) * INPUT_LAYER_SIZE) * sizeof(float));
    cudaDeviceSynchronize();
    double start_compute_time = get_time();
    // normalise the inputs
    dim3 threadperBlockNorm(BLOCK_SIZE * BLOCK_SIZE);
    dim3 blocksperGridNorm((PADDED_BATCH_SIZE * INPUT_LAYER_SIZE + threadperBlockNorm.x - 1) / threadperBlockNorm.x);
    normalise_kernel<<<blocksperGridNorm, threadperBlockNorm>>>(d_input_layer_1, PADDED_BATCH_SIZE * INPUT_LAYER_SIZE);
    CHECK_CUDA_ERROR(cudaPeekAtLastError());
    // forward_pass
    forward_pass(handle);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    double end_compute_time = get_time();
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_layer_2, d_output_layer_2, PADDED_BATCH_SIZE * OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    double end_inference_time = get_time();

    int final_prediction[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        float output = h_output_layer_2[i * OUTPUT_LAYER_SIZE + 0];
        float idx = 0;
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++)
        {
            if (h_output_layer_2[i * OUTPUT_LAYER_SIZE + j] > output)
            {
                output = h_output_layer_2[i * OUTPUT_LAYER_SIZE + j];
                idx = j;
            }
        }
        final_prediction[i] = (int)idx;
    }
    free_memory();

    float real_values[BATCH_SIZE];
    printf("Total Inference Time: %f ms\n", (end_inference_time - start_inference_time) * 1000);
    printf("Total Compute Time: %f ms\n", (end_compute_time - start_compute_time) * 1000);

    printf("Starting accuracy calculation...\n");
    read_file((char *)"binary_files/labels.bin", real_values, BATCH_SIZE);
    int correct_predictions = 0;
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        if (final_prediction[i] == (int)real_values[i])
            correct_predictions++;
    }
    float accuracy = (float)correct_predictions / BATCH_SIZE * 100.0f;
    printf("Accuracy: %.2f%%\n", accuracy);

    return 0;
}