#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

void PrintMatrixf16(__half* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", __half2float(arr[i * cols + j]));
		printf("\n");
	}
	printf("\n");
}

__global__ void CurandNormalizef16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * range + min);
}

void CurandGenerateUniformf16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizef16 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 0.0000152590218967f);
}

int main()
{
	const uint32_t INPUTS = 100000000;

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 0);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;

	__half* input = new __half[INPUTS];
	__half* inputGPU;
	cudaMalloc(&inputGPU, INPUTS * sizeof(__half));

	cudaEventRecord(start);
	for (uint32_t i = 10; i--;)
		CurandGenerateUniformf16(curandGenerator, inputGPU, INPUTS);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken: %f ms\n", milliseconds);

	cudaMemcpy(input, inputGPU, INPUTS * sizeof(__half), cudaMemcpyDeviceToHost);
	//PrintMatrixf16(input, INPUTS, 1, "Input");

	uint32_t arr[100] = {};
	for (uint32_t i = INPUTS; i--;)
	{
		arr[uint32_t(__half2float(input[i]) * 40 + 50)]++;
	}

	printf("Histogram:\n");
	for (uint32_t i = 100; i--;)
	{
		for (uint32_t j = arr[i] * 1000 / INPUTS; j--;)
			printf("*");
		printf("\n");
	}
	printf("Histogram End");

	return 0;
}