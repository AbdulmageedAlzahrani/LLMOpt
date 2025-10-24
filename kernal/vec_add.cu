#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1<<20;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + block.x - 1) / block.x);
    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    printf("c[0]=%.1f  c[N-1]=%.1f\n", h_c[0], h_c[N-1]);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
