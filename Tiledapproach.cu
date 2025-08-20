// This code evaluates matric multiplication using tiled approach

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
using std::cout;


__global__ void matrixMulTiled(float* A, float* B, float* C, int N) {

	// Define shared memory for tiles
	__shared__ float tile_A[16][16];
	__shared__ float tile_B[16][16];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f;

	// Loop over tiles
	for (int t = 0; t < (N + 15) / 16; t++) {
		if (row < N && t * 16 + threadIdx.x < N) {
			tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
		}
		else {
			tile_A[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (col < N && t * 16 + threadIdx.y < N) {
			tile_B[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
		}
		else {
			tile_B[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (int k = 0; k < 16; k++) {
			sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < N && col < N) {
		C[row * N + col] = sum;
	}
}

void verifyResult(const float* A, const float* B, const float* C, int N) {

	for (int i = 0; i < N * N; i++) {
		float expected = 0.0f;
		for (int k = 0; k < N; k++) {
			expected += A[i / N * N + k] * B[k * N + i % N];
		}
		if (fabs(C[i] - expected) > 1e-3) {
			cout << "Mismatch at index " << i << ": expected " << expected << ", got " << C[i] << "\n";
			return;
		}
	}
	cout << "All results are correct!\n";
}

int main() {

	// Define size of the matrix
	int N = 2048; // Size of the matrix

	size_t bytes = N * N * sizeof(float);

	// Define memory on host
	float* h_A, * h_B, * h_C;

	// Allocate memory on host
	h_A = (float*)malloc(bytes);
	h_B = (float*)malloc(bytes);
	h_C = (float*)malloc(bytes);

	// Initialize matrices
	for (int i = 0; i < N * N; i++) {
		h_A[i] = static_cast<float>(i % 100);
		h_B[i] = static_cast<float>((i + 1) % 100);
		h_C[i] = 0.0f; // Initialize C to zero
	}

	// Define pointers for device memory
	float* d_A, * d_B, * d_C;

	// Allocate memory on device
	cudaMalloc((void**)&d_A, bytes);
	cudaMalloc((void**)&d_B, bytes);
	cudaMalloc((void**)&d_C, bytes);

	// Copy matrices from host to device
	cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

	// Define grid and block dimensions
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Start timer
	auto start = std::chrono::high_resolution_clock::now();
	// Launch the kernel
	matrixMulTiled <<< blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	// Stop timer
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	cout << "Kernel execution time for Tiled approach: " << duration.count() << " ms\n";

	// Copy result back to host
	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	// Verify the result
	verifyResult(h_A, h_B, h_C, N);

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;

}