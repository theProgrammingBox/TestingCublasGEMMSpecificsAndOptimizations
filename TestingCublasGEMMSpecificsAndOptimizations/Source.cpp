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

__global__ void CurandNormalizef16v2(__half* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		__half range = __float2half(0.0000152590218967f);
		uint16_t outputBits = *(uint16_t*)(output + index);
		uint16_t rangeBits = *(uint16_t*)&range;
		uint16_t rangeMantissa = rangeBits & 0x3FF | 0x400;
		uint16_t rangeExponent = rangeBits & 0x7c00;
		uint32_t resultMantissa = 0;
		while (outputBits)
		{
			resultMantissa += (outputBits & 1) * rangeMantissa;
			outputBits >>= 1;
			rangeMantissa <<= 1;
		}
		resultMantissa >>= 10;
		while (resultMantissa & 0xf800)
		{
			resultMantissa >>= 1;
			rangeExponent++;
		}
		if (resultMantissa & 0x3ff)
		{
			while (~resultMantissa & 0x400)
			{
				resultMantissa <<= 1;
				rangeExponent--;
			}
		}
		uint16_t result = rangeBits & 0x8000 | rangeExponent | resultMantissa;
		output[index] = *(__half*)&result;
	}
}

void CurandGenerateUniformf16v2(curandGenerator_t generator, __half* output, uint32_t size)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizef16v2 <<<std::ceil(0.0009765625f * size), 1024 >>> (output, size);
}

int main()
{
	__half range = __float2half(0.0000152590218967f);
	uint16_t outputBits = 1;// 0xffff;
	uint16_t rangeBits = *(uint16_t*)&range;
	uint16_t rangeMantissa = rangeBits & 0x3FF | 0x400;
	uint16_t rangeExponent = rangeBits & 0x7c00;
	uint32_t resultMantissa = 0;

	for (int32_t i = 15; i >= 0; i--)
	{
		printf("%d", (rangeBits >> i) & 1);
		if (i == 15 || i == 10) printf(" ");
	}
	printf("\n");

	for (int32_t i = 15; i >= 0; i--)
	{
		printf("%d", (rangeMantissa >> i) & 1);
		if (i == 15 || i == 10) printf(" ");
	}
	printf("\n");

	for (int32_t i = 15; i >= 0; i--)
	{
		printf("%d", (rangeExponent >> i) & 1);
		if (i == 15 || i == 10) printf(" ");
	}
	printf("\n");

	while (outputBits)
	{
		resultMantissa += (outputBits & 1) * rangeMantissa;
		outputBits >>= 1;
		rangeMantissa <<= 1;
	}

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (resultMantissa >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");

	resultMantissa >>= 10;

	for (int32_t i = 31; i >= 0; i--)
	{
		printf("%d", (resultMantissa >> i) & 1);
		if (i == 31 || i == 23) printf(" ");
	}
	printf("\n");

	while (resultMantissa & 0xf800)
	{
		resultMantissa >>= 1;
		rangeExponent += 0x400;
	}
	if (resultMantissa & 0x3ff)
	{
		while (~resultMantissa & 0x400)
		{
			resultMantissa <<= 1;
			rangeExponent -= 0x400;
		}
	}

	for (int32_t i = 15; i >= 0; i--)
	{
		printf("%d", (rangeExponent >> i) & 1);
		if (i == 15 || i == 10) printf(" ");
	}
	printf("\n");

	rangeExponent ^= 0x4000;
	rangeExponent += 0x400;

	uint16_t result = rangeBits & 0x8000 | (rangeExponent & 0x7c00) | (resultMantissa & 0x3ff);

	for (int32_t i = 15; i >= 0; i--)
	{
		printf("%d", (result >> i) & 1);
		if (i == 15 || i == 10) printf(" ");
	}
	printf("\n");

	__half out = *(__half*)&result;
	printf("%f\n", __half2float(out));
	return 0;

	const uint32_t INPUTS = 20;// 100000000;

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
	for (uint32_t i = 1; i--;)
		//CurandGenerateUniformf16(curandGenerator, inputGPU, INPUTS);
		CurandGenerateUniformf16v2(curandGenerator, inputGPU, INPUTS);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken: %f ms\n", milliseconds);

	cudaMemcpy(input, inputGPU, INPUTS * sizeof(__half), cudaMemcpyDeviceToHost);
	PrintMatrixf16(input, INPUTS, 1, "Input");

	/*uint32_t arr[100] = {};
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
	printf("Histogram End");*/

	return 0;
}