

#include <iostream>
#include<cuda_runtime.h>
using std::cout;

__global__  void matrixMul(int* A, int* B, int* C, int N) {

	// Calculate the row and column index for the current thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Check if the thread is within the bounds of the matrix
	if (row < N && col < N) {
		int sum = 0;
		for (int k = 0; k < N; k++) {
			sum += A[row * N + k] * B[k * N + col];
		}
		C[row * N + col] = sum;
	}

}

void verifyresult(int* h_A, int* h_B, int* h_C, int N) {
	for (int i = 0; i < N * N; i++) {
		int expected = 0;
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
	int* h_A, * h_B, * h_C;

	int N = 1<<10; // Size of the vectors

	// define size of a matrix
	size_t bytes = N * N * sizeof(int);


	// allocate memory on host
	h_A = (int*)malloc(bytes);
	h_B = (int*)malloc(bytes);
	h_C = (int*)malloc(bytes);

	// Define pointers for device memory
	int* d_A, * d_B, * d_C;

	// Allocate memory on device
	cudaMalloc((void**)&d_A, bytes);
	cudaMalloc((void**)&d_B, bytes);
	cudaMalloc((void**)&d_C, bytes);

	// Initialize host matrices
	for (int i = 0; i < N * N; i++) {
		h_A[i] = i % 10; // Example initialization
		h_B[i] = (i + 1) % 10; // Example initialization
		h_C[i] = 0; // Initialize result matrix to zero
	}

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

	// Call kernel function
	
	matrixMul<<<grid, block>>>(d_A, d_B, d_C, N);

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