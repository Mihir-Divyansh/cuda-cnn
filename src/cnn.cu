#include "matrix.cu"

#define tidx threadIdx.x
#define tidy threadIdx.y
#define tidz threadIdx.z
#define bid  blockIdx.x
#define bdim blockDim.x

__device__
void printMatrix(const float* mat, size_t rows, size_t cols, size_t depth) {
    if (tidx == 0 && tidy == 0) {
        for (size_t d = 0; d < depth; ++d) {
            printf("Depth %zu (Block %d):\n", d, bid);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    size_t idx = (i * cols + j) * depth + d;
                    printf("%f ", mat[idx]);
                }
                printf("\n");
            }
        }
        printf("\n");
    }
}
__global__ 
void convFilter (dMatrix data, dMatrix filter, float* __restrict__ output, size_t stride)
{
	if (data.depth != filter.depth)
	{
		printf("Dimension mismatch between Data and Filter!\n");
		return;
	}
	extern 	__shared__ float filtermat[];
	size_t thread_id = threadIdx.z * blockDim.y * blockDim.x 
                 + threadIdx.y * blockDim.x 
                 + threadIdx.x;
	for (size_t i = thread_id; i < filter.size(); i += blockDim.x * blockDim.y * blockDim.z) 
	{
    		filtermat[i] = filter.dmat[i];
	}


	//if (tidx == 0 && bid == 0) printMatrix(data.dmat, data.rows, data.cols, 1);	
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t k = blockIdx.z * blockDim.z + threadIdx.z;
	size_t rows_out = ((data.rows - filter.rows) / stride) + 1;
	size_t cols_out = ((data.cols - filter.cols) / stride) + 1;

	if (i < cols_out && j < rows_out)
	{
		float sum = 0.f;
		#pragma unroll
		for(int dx = 0; dx < filter.rows; dx++)
		{
			#pragma unroll
			for (int dy = 0; dy < filter.cols; dy++)
			{
				#pragma unroll
				for (size_t dz = 0; dz < filter.depth; dz ++)	
				{
					sum += 1.f / filter.depth * data.dmat[ (i * stride + dx) * data.cols * filter.depth + (j * stride + dy) * filter.depth + dz ] * filtermat[dx * filter.cols * filter.depth + dy * filter.depth + dz];
				}
			}
		}
		//printf("Thread : %d, Block : %d\n", threadIdx.x, blockIdx.x);
		
		output[i * cols_out + j] =  sum;
		//printf(" sum = %f\n",sum);
	}
}
