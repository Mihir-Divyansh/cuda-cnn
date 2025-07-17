#include "matrix.cu"
#include <cstddef>
#include <time.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <math.h>
#include "parsepm.c"

__global__ void convFilter(dMatrix input, dMatrix filter, float* output, size_t stride);
unsigned char *read_ppm(const char*, int*, int*);
void savePGM(float* data, size_t width, size_t height, char* filename); 

void printMatrix3D(const float* mat, size_t rows, size_t cols, size_t depth, const char* name) {
	printf("\n%s [depth-major slices]:\n", name);
	for (size_t d = 0; d < depth; ++d) {
		printf("\nDepth %zu:\n", d);
		for (size_t i = 0; i < rows; ++i) {
			for (size_t j = 0; j < cols; ++j) {
				size_t idx = (i * cols + j) * depth + d;
				printf("%g ", mat[idx]);
			}
			printf("\n");
		}
	}
}

// Define different image processing kernels
void loadKernel(Matrix& filter, const char* kernelType) {
    size_t f_rows = filter.rows;
    size_t f_cols = filter.cols;
    size_t f_depth = filter.depth;
    
    if (strcmp(kernelType, "blur") == 0) {
        // Box blur kernel (3x3)
        float blur_kernel[] = {
            1.0f/9, 1.0f/9, 1.0f/9,
            1.0f/9, 1.0f/9, 1.0f/9,
            1.0f/9, 1.0f/9, 1.0f/9
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = blur_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "gaussian") == 0) {
        // Gaussian blur kernel (3x3)
        float gaussian_kernel[] = {
            1.0f/16, 2.0f/16, 1.0f/16,
            2.0f/16, 4.0f/16, 2.0f/16,
            1.0f/16, 2.0f/16, 1.0f/16
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = gaussian_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "sharpen") == 0) {
        // Sharpening kernel (3x3)
        float sharpen_kernel[] = {
             0.0f, -1.0f,  0.0f,
            -1.0f,  5.0f, -1.0f,
             0.0f, -1.0f,  0.0f
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = 1.0 / 9 * sharpen_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "edge_sobel_x") == 0) {
        // Sobel X edge detection kernel (3x3)
        float sobel_x_kernel[] = {
            -1.0f,  0.0f,  1.0f,
            -2.0f,  0.0f,  2.0f,
            -1.0f,  0.0f,  1.0f
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = 1.0 / 9 * sobel_x_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "edge_sobel_y") == 0) {
        // Sobel Y edge detection kernel (3x3)
        float sobel_y_kernel[] = {
            -1.0f, -2.0f, -1.0f,
             0.0f,  0.0f,  0.0f,
             1.0f,  2.0f,  1.0f
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = 1.0 / 9 * sobel_y_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "edge_laplacian") == 0) {
        // Laplacian edge detection kernel (3x3)
        float laplacian_kernel[] = {
             0.0f, -1.0f,  0.0f,
            -1.0f,  4.0f, -1.0f,
             0.0f, -1.0f,  0.0f
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = 1.0/9 * laplacian_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "emboss") == 0) {
        // Emboss kernel (3x3)
        float emboss_kernel[] = {
            -2.0f, -1.0f,  0.0f,
            -1.0f,  1.0f,  1.0f,
             0.0f,  1.0f,  2.0f
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = 1.0 / 9 * emboss_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "motion_blur") == 0) {
        // Motion blur kernel (3x3) - diagonal
        float motion_kernel[] = {
            1.0f/3, 0.0f,    0.0f,
            0.0f,   1.0f/3,  0.0f,
            0.0f,   0.0f,    1.0f/3
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = motion_kernel[i];
            }
        }
    }
    else if (strcmp(kernelType, "ridge") == 0) {
        // Ridge detection kernel (3x3)
        float ridge_kernel[] = {
            -1.0f, -1.0f, -1.0f,
            -1.0f,  8.0f, -1.0f,
            -1.0f, -1.0f, -1.0f
        };
        for (size_t d = 0; d < f_depth; ++d) {
            for (size_t i = 0; i < f_rows * f_cols; ++i) {
                filter.mat[i * f_depth + d] = 1.0 / 9 * ridge_kernel[i];
            }
        }
    }
    else {
        printf("Unknown kernel type: %s. Using default blur kernel.\n", kernelType);
        loadKernel(filter, "blur");
    }
}

void processWithKernel(Matrix& input, Matrix& output, const char* kernelType, 
                      size_t stride, const char* outputFilename) {
    const size_t f_rows = 3, f_cols = 3, f_depth = 3;
    const size_t out_rows = (input.rows - f_rows) / stride + 1;
    const size_t out_cols = (input.cols - f_cols) / stride + 1;
    
    cudaEvent_t start, stop;
    float h2d_time = 0, kernel_time = 0, d2h_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Create and load the filter
    Matrix filter(f_rows, f_cols, f_depth);
    loadKernel(filter, kernelType);
    
    printf("\n=== Processing with %s kernel ===\n", kernelType);
    printMatrix3D(filter.mat, f_rows, f_cols, f_depth, "convFilter");
    
    // Allocate device memory
    dMatrix d_input(input.rows, input.cols, input.depth);
    dMatrix d_filter(f_rows, f_cols, f_depth);
    float *d_output;
    cudaMalloc(&d_output, out_cols * out_rows * sizeof(float));
    
    // Copy input -> device
    cudaEventRecord(start);
    input.copyToDevice(d_input, input.size());
    filter.copyToDevice(d_filter, filter.size());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);
    
    // Launch kernel
    dim3 block(4, 4, 4); 
    dim3 grid((out_cols + block.x - 1) / block.x,
              (out_rows + block.y - 1) / block.y,
              (input.depth + block.z - 1) / block.z);
    
    cudaEventRecord(start);
    convFilter<<<grid, block, filter.size()*sizeof(float)>>>(d_input, d_filter, d_output, stride);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);
    
    // Copy output back to host
    cudaEventRecord(start);
    cudaMemcpy(output.mat, d_output, out_cols * out_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);
    
    // Save output image
    savePGM(output.mat, out_cols, out_rows, const_cast<char*>(outputFilename));
    
    // Report timings
    printf("Host to Device Copy Time: %.3f ms\n", h2d_time);
    printf("Kernel Execution Time:    %.3f ms\n", kernel_time);
    printf("Device to Host Copy Time: %.3f ms\n", d2h_time);
    printf("Total Processing Time:    %.3f ms\n", h2d_time + kernel_time + d2h_time);
    printf("Output saved to: %s\n", outputFilename);
    
    // Cleanup
    d_input.freeMat();
    d_filter.freeMat();
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const size_t in_rows = 99, in_cols = 99, in_depth = 3;
    const size_t f_rows = 3, f_cols = 3, f_depth = 3;
    const size_t stride = 1;
    const size_t out_rows = (in_rows - f_rows) / stride + 1;
    const size_t out_cols = (in_cols - f_cols) / stride + 1;
    
    unsigned char *raw_data;
    int w, h;
    raw_data = read_ppm("imgs/sign_1.ppm", &w, &h);
    
    // Allocate host memory
    Matrix input(in_rows, in_cols, in_depth);
    Matrix output(out_rows, out_cols, 1);
    
    // Fill input with image data
    for (size_t i = 0; i < input.size(); ++i) {
        input.mat[i] = (float)(raw_data[i]);
    }

	savePGM(input.mat, in_cols, in_rows, "rgb_test");
    
    printf("Input image loaded: %dx%dx%d\n", (int)in_rows, (int)in_cols, (int)in_depth);
    
    // Array of kernels to test
    const char* kernels[] = {
        "blur",
        "gaussian", 
        "sharpen",
        "edge_sobel_x",
        "edge_sobel_y",
        "edge_laplacian",
        "emboss",
        "motion_blur",
        "ridge"
    };
    
    const char* output_files[] = {
        "imgs/sign_1_blur.pgm",
        "imgs/sign_1_gaussian.pgm",
        "imgs/sign_1_sharpen.pgm",
        "imgs/sign_1_sobel_x.pgm",
        "imgs/sign_1_sobel_y.pgm",
        "imgs/sign_1_laplacian.pgm",
        "imgs/sign_1_emboss.pgm",
        "imgs/sign_1_motion.pgm",
        "imgs/sign_1_ridge.pgm"
    };
    
    int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
    
    // Process image with each kernel
    for (int i = 0; i < num_kernels; ++i) {
        processWithKernel(input, output, kernels[i], stride, output_files[i]);
        printf("\n-----------------------------------------------------------------------------------\n");
    }
    
    printf("\n=== All kernels processed successfully! ===\n");
    printf("Check the 'imgs/' directory for output files.\n");
    
    return 0;
}
