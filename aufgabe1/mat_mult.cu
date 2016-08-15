#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include "../aufgabe2/ca_common.h"

#define BLOCK_SIZE 32
#define TILE_WIDTH BLOCK_SIZE

#define M(m, row, col) m.elements[(row) * m.cols + (col)]
#define SIZE(m) m.cols * m.rows * sizeof(m_cell)

#ifdef TILING
#define MULT_KERNEL mat_mult_tiling_kernel
#else
#define MULT_KERNEL mat_mult_kernel
#endif

#define m_cell_fms "%d"
typedef int m_cell;

typedef struct { 
    int rows;
    int cols; 
    m_cell* elements;
} Matrix;

bool matrix_equal(const Matrix a, const Matrix b);
void mat_mult(const Matrix, const Matrix, const Matrix);
void print_matrix(const Matrix);
__global__ void mat_mult_kernel(const Matrix a, const Matrix b, const Matrix c);
__global__ void mat_mult_tiling_kernel(const Matrix a, const Matrix b, const Matrix c);
bool set_cache_config(int cache_config, const char ** cache_config_str);
void mat_mult_init(int argc, char** argv, int *n, int *cache_config, const char ** cache_config_str);

int main(int argc, char** argv)
{
    int n, cache_config;
    const char *cache_config_str;
    
    mat_mult_init(argc, argv, &n, &cache_config, &cache_config_str);

    Matrix a, b, c;
    Matrix d_a, d_b, d_c;
 
    a.cols = a.rows = n;
    b.cols = b.rows = n;

    c.rows = a.rows;
    c.cols = b.cols;

    MALLOC_ERROR_CHECK(a.elements = (m_cell*) malloc(SIZE(a)));
    MALLOC_ERROR_CHECK(b.elements = (m_cell*) malloc(SIZE(b)));
    MALLOC_ERROR_CHECK(c.elements = (m_cell*) malloc(SIZE(c)));

    // c_host matrix calculated locally (compared with c matrix for validation) 
    Matrix c_host;
    c_host.rows = c.rows;
    c_host.cols = c.cols;
    MALLOC_ERROR_CHECK(c_host.elements = (m_cell*) malloc(SIZE(c_host)));

    int row;
    int col;
    for (row = 0; row < n; row++)
    {
        for(col = 0; col < n; col++)
        {
            M(a, row, col) = row * n + col;
            M(b, col, row) = row * n + col;
        }
    }

    //print_matrix(a);
    //print_matrix(b);

    d_a.cols = a.cols;
    d_a.rows = a.rows;
    cudaMalloc(&d_a.elements,SIZE(d_a));
    cudaMemcpy(d_a.elements, a.elements, SIZE(d_a), cudaMemcpyHostToDevice);

    d_b.cols = b.cols;
    d_b.rows = b.rows;
    cudaMalloc(&d_b.elements, SIZE(d_b));
    cudaMemcpy(d_b.elements, b.elements, SIZE(d_b), cudaMemcpyHostToDevice);

    d_c.cols = c.cols;
    d_c.rows = c.rows;
    cudaMalloc(&d_c.elements, SIZE(d_c));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(c.cols / dimBlock.x, c.rows / dimBlock.y);

        
    TIME_GET(start);
    MULT_KERNEL<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    TIME_GET(stop);
    
    double kernel_time = TIME_DIFF(start, stop);
   
    CUDA_ERROR_CHECK(cudaPeekAtLastError());

    cudaMemcpy(c.elements, d_c.elements, SIZE(d_c), cudaMemcpyDeviceToHost);

    //print_matrix(c);

    mat_mult(a ,b ,c_host);
    // validate c (kernel result matrix) with c_host matrix
    bool valid_result = matrix_equal(c, c_host);
  
    // free memory
    free(a.elements);
    free(b.elements);
    free(c.elements);
    free(c_host.elements);

    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    cudaFree(d_c.elements);

    if (valid_result)
    {
        printf("{ \"valid\": true, \"n\": %d, \"kernel_time\": %.9f, \"cache_config\": \"%s\", \"kernel\": \"%s\"}\n", n, kernel_time, cache_config_str, STR(MULT_KERNEL));
        return EXIT_SUCCESS;
    }
    else
    { 
        printf("{\"valid\": false}\n");
        return EXIT_FAILURE;
    }
}

void mat_mult_init(int argc, char** argv, int *n, int *cache_config, const char ** cache_config_str)
{
	assert(argc == 3);

	*n = atoi(argv[1]);
	*cache_config = atoi(argv[2]);

	assert((*n > 0) && (*n % BLOCK_SIZE == 0));
        assert(set_cache_config(*cache_config, cache_config_str));
}

bool set_cache_config(int cache_config, const char ** cache_config_str)
{
    bool success = true;
    
    switch(cache_config)
    {
        case cudaFuncCachePreferNone:
            CUDA_ERROR_CHECK(cudaFuncSetCacheConfig(MULT_KERNEL, cudaFuncCachePreferNone));
            *cache_config_str = "prefer_none";
            break;
        case cudaFuncCachePreferShared:
            CUDA_ERROR_CHECK(cudaFuncSetCacheConfig(MULT_KERNEL, cudaFuncCachePreferShared));
            *cache_config_str = "prefer_shared";
            break; 
        case cudaFuncCachePreferL1:
            CUDA_ERROR_CHECK(cudaFuncSetCacheConfig(MULT_KERNEL, cudaFuncCachePreferL1));
            *cache_config_str = "prefer_L1";
            break; 
        case cudaFuncCachePreferEqual:
            CUDA_ERROR_CHECK(cudaFuncSetCacheConfig(MULT_KERNEL, cudaFuncCachePreferEqual));
            *cache_config_str = "prefer_equal";
            break;
        default :
            *cache_config_str = "undefined";
            success = false;
    }

    return success;
}

bool matrix_equal(const Matrix a, const Matrix b)
{
    if (a.cols != b.cols || a.rows != b.rows)
    {
        return false;
    } 
    
    int row;
    int col;
    for (row = 0; row < a.rows; row++)
    {
        for(col = 0; col < a.cols; col++)
        {
            if (M(a, row, col) != M(b, row, col))
            {
                return false;
            } 
        }

    }
    
    return true;
}

void mat_mult(const Matrix a, const Matrix b, const Matrix c)
{
    int row;
    int col;
    for (row = 0; row < c.rows; row++)
    {
        for(col = 0; col < c.cols; col++)
        {
            int inner;
            m_cell sum = 0;
            for (inner = 0; inner < a.cols; inner++)
            {
                sum += M(a, row, inner) * M(b, inner, col);
            }
            M(c, row, col) = sum; 
        }
    }
}

__global__ void mat_mult_kernel(const Matrix a, const Matrix b, const Matrix c) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int inner;
    m_cell sum = 0;
    for (inner = 0; inner < a.cols; inner++)
    {
        sum += M(a, row, inner) * M(b, inner, col);
    }
    M(c, row, col) = sum; 
}

__global__ void mat_mult_tiling_kernel(const Matrix a, const Matrix b, const Matrix c)
{
    __shared__ m_cell a_ds [TILE_WIDTH][TILE_WIDTH];
    __shared__ m_cell b_ds [TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    m_cell cval = 0;

    int t;
    for (t = 0; t < a.cols/TILE_WIDTH; t++)
    {
        a_ds[ty][tx] = M(a, row, t*TILE_WIDTH+tx);
        b_ds[ty][tx] = M(b, t*TILE_WIDTH + ty, col);
        
        __syncthreads();
        
        int i; 
        for(i=0; i < TILE_WIDTH; i++)
        {
            cval += a_ds[ty][i] * b_ds[i][tx];
        }
        
        __syncthreads();
    }

    M(c, row, col) = cval;
}

void print_matrix(const Matrix m)
{
    int row;
    int col;
    for (row = 0; row < m.rows; row++)
    {
        for(col = 0; col < m.cols; col++)
        {
            printf("\t"m_cell_fms, M(m, row, col));
        }
        printf("\n");
    }
    printf("\n\n");
}

