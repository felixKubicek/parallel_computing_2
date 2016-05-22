#include <stdio.h>
#include <stdlib.h> 

#define N 5
#define m_cell_fms "%d"
#define M(m, row, col) m.elements[row * m.rows + col]

typedef int m_cell;

typedef struct { 
  int rows; 
  int cols; 
  m_cell* elements;
} Matrix;


void mat_mult(const Matrix, const Matrix, const Matrix);
void print_matrix(const Matrix);
__global__ void mat_mult_kernel(const Matrix, const Matrix, const Matrix);

int main(void)
{
    Matrix a, b, c;
    //Matrix d_a, d_b, d_c;

    a.cols = a.rows = N;
    b.cols = b.rows = N;

    c.rows = a.rows;
    c.cols = b.cols;

    // TODO: catch errors
    a.elements = (m_cell*) calloc(a.cols*a.rows, sizeof(m_cell));
    b.elements = (m_cell*) calloc(b.cols*b.rows, sizeof(m_cell));
    c.elements = (m_cell*) calloc(c.cols*c.rows, sizeof(m_cell));

    int row;
    int col;
    for (row = 0; row < N; row++)
    {
        for(col = 0; col < N; col++)
        {
            M(a, row, col) = row * N + col;
            M(b, col, row) = row * N + col;
        }
    }


    /* 
    d_a.cols = a.cols;
    d_a.rows = a.rows;
    size_t size = d_a.cols * d_a.rows * sizeof(m_cell);
    cudaMalloc(&d_a.elements, size);
    cudaMemcpy(d_a.elements, a.elements, size, cudaMemcpyHostToDevice);

    d_b.cols = b.cols;
    d_b.rows = b.rows;
    size_t size = d_b.cols * d_b.rows * sizeof(m_cell);
    cudaMalloc(&d_b.elements, size);
    cudaMemcpy(d_b.elements, b.elements, size, cudaMemcpyHostToDevice);

    d_c.cols = c.cols;
    d_c.rows = c.rows;
    size_t size = d_c.cols * d_c.rows * sizeof(m_cell);
    cudaMalloc(&d_c.elements, size);
 
    cudaMemcpy(c.elements, d_c.elements, size, cudaMemcpyDeviceToHost);
    */

    print_matrix(a);
    print_matrix(b);
    mat_mult(a, b, c);
    print_matrix(c);


    // free memory
    free(a.elements);
    free(b.elements);
    free(c.elements);

    //cudaFree(d_a.elements);
    //cudaFree(d_b.elements);
    //cudaFree(d_c.elements);

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
