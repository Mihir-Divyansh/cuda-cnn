#include <cstddef>
#include <stdio.h>
#include "cuda_runtime.h"

#define HOST_MALLOC_ERROR -1
#define HOST_MALLOC_INIT_SUCCESS 0
#define DEVICE_MALLOC_ERROR -2
struct dMatrix
{
	size_t rows, cols, depth;
	cudaError_t err;
	float * dmat;

	__host__
	dMatrix(size_t num_rows, size_t num_cols, size_t mat_depth)
		: rows(num_rows), cols(num_cols), depth(mat_depth), dmat(nullptr)
	{
		err = cudaMalloc(&dmat, size() * sizeof(float));
		if (err != cudaSuccess || dmat == nullptr) {
			dmat = nullptr;
		}
	}

	__host__ __device__
	inline size_t size() const {
		return rows * cols * depth;
	}

	__host__
	int freeMat()
	{
		if (dmat != nullptr) cudaFree(dmat);
		dmat = nullptr;
		rows = cols = depth = 1;
		return 0;
	}

	__host__
	~dMatrix() {
		freeMat();
	}
	
};

struct Matrix
{
	size_t rows, cols, depth;
	int status;
	cudaError_t err;
	float * mat;

	__host__
	Matrix(size_t num_rows, size_t num_cols, size_t mat_depth)
		: rows(num_rows), cols(num_cols), depth(mat_depth), mat(nullptr)
	{
		err = cudaHostAlloc(&mat, size() * sizeof(float), 0);
		if (err != cudaSuccess || mat == nullptr) {
			status = HOST_MALLOC_ERROR;
			mat = nullptr;
		} else {
			status = HOST_MALLOC_INIT_SUCCESS;
		}
	}

	__host__ __device__
	inline size_t size() const {
		return rows * cols * depth;
	}

	__host__
	inline int cpyHH(float *src, size_t nelems)
	{
		memcpy(mat, src, nelems * sizeof(float));
		return 0;
	}

	__host__
	int freeMat()
	{
		if (mat != nullptr) cudaFreeHost(mat);
		mat = nullptr;
		rows = cols = depth = 1;
		return 0;
	}

	__host__
	~Matrix() {
		freeMat();
	}
	__host__
	inline cudaError_t copyToDevice(dMatrix &deviceMat, size_t nelems, size_t offset = 0) const
	{
		return cudaMemcpy(deviceMat.dmat + offset, mat + offset, nelems * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	__host__
	inline cudaError_t copyFromDevice(const dMatrix &deviceMat, size_t nelems, size_t offset = 0)
	{
		return cudaMemcpy(mat + offset, deviceMat.dmat + offset, nelems * sizeof(float), cudaMemcpyDeviceToHost);
	}

	__host__
	inline cudaError_t copyToDeviceAsync(dMatrix &deviceMat, size_t nelems, size_t offset, cudaStream_t stream) const
	{
		return cudaMemcpyAsync(deviceMat.dmat + offset, mat + offset, nelems * sizeof(float), cudaMemcpyHostToDevice, stream);
	}

	__host__
	inline cudaError_t copyFromDeviceAsync(const dMatrix &deviceMat, size_t nelems, size_t offset, cudaStream_t stream)
	{
		return cudaMemcpyAsync(mat + offset, deviceMat.dmat + offset, nelems * sizeof(float), cudaMemcpyDeviceToHost, stream);
	}

};



