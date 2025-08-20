

#include <iostream>
#include<cuda_runtime.h>
#include<chrono>
using std::cout;

__global__ void matrixMul(float* A, float* B, float* C, int N) {

	// Calculate the row and column index for the current thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the thread is within the bounds of the matrix
	if (row < N && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < N; k++) {
			sum += A[row * N + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}
}

void verifyresult(float* h_A, float* h_B, float* h_C, int N) {
	for (int i = 0; i < N * N; i++) {
		float expected = 0.0f;
		for (int k = 0; k < N; k++) {
			expected += h_A[i / N * N + k] * h_B[k * N + i % N];
		}
		if (h_C[i] != expected) {
			cout << "Mismatch at index " << i << ": expected " << expected << ", got " << h_C[i] << "\n";
			return;
		}
	}
	cout << "All results are correct!\n";
}


int main(){

	// define memory on host
	float* h_A, * h_B, * h_C;

	int N = 2048; // Size of the vectors

	// define size of a matrix
	size_t bytes = N * N * sizeof(float);


	// allocate memory on host
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
	// Define thread per block
	
	int threadsPerBlock = 16;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	// Launch the kernel
	dim3 block(threadsPerBlock, threadsPerBlock);
	dim3 grid(blocksPerGrid, blocksPerGrid);

	//Start timer
	auto start = std::chrono::high_resolution_clock::now();
	
	// Call kernel function
	matrixMul<<<grid, block>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();

	// Stop timer
	auto end = std::chrono::high_resolution_clock::now();

	// Calculate elapsed time
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	cout << "Kernel execution time for baseline: " << duration.count() << " ms\n";


	// Copy result matrix from device to host
	cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

	// Verify the result
	verifyresult(h_A, h_B, h_C, N);

	cout << "Completed Successfully";

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}