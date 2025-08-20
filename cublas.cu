
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>

using std::cout;


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
	int N = 1 << 10; // Size of the matrix

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
	cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);

	// Create cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 0.0f;

	// Start timing
	auto start = std::chrono::high_resolution_clock::now();

	// Perform matrix multiplication using cuBLAS
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);

	// Stop timing
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate elapsed time
	std::chrono::duration<double, std::milli> elapsed = end - start;

	cout << "Time taken for cuBLAS matrix multiplication: " << elapsed.count() << " ms\n";

	// Copy result back to host
	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	// Verify result
	verifyResult(h_A, h_B, h_C, N);

	// Clean up
	cublasDestroy(handle);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);


	return 0;
}

	
