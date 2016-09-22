/*
 * 2016 Felix Kubicek
 *
 * Matrix multiplication with and without tiling.
 * Both versions are taken from the cuda programming guide (http://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory).
 *
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include "openssl/md5.h"
#include "mat_mult.h"
#include "../aufgabe2/ca_common.h"

void mat_mult(const Matrix, const Matrix, const Matrix);
void mat_mult_init(int argc, char** argv, int *n);

int main(int argc, char** argv)
{
    int n;

    mat_mult_init(argc, argv, &n);

    Matrix a, b, c;
 
    a.cols = a.rows = n;
    b.cols = b.rows = n;

    c.rows = a.rows;
    c.cols = b.cols;

    MALLOC_ERROR_CHECK(a.elements = (m_cell*) malloc(SIZE(a)));
    MALLOC_ERROR_CHECK(b.elements = (m_cell*) malloc(SIZE(b)));
    MALLOC_ERROR_CHECK(c.elements = (m_cell*) malloc(SIZE(c)));

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

    TIME_GET(start);
    mat_mult(a ,b ,c);
    TIME_GET(stop);
    
    double seq_time = TIME_DIFF(start, stop);

    uint8_t hash[MD5_DIGEST_LENGTH];
    MD5_CTX ctx;
    MD5_Init(&ctx);
    MD5_Update(&ctx, c.elements, SIZE(c));
    MD5_Final(hash, &ctx);
    char* hash_str = ca_buffer_to_hex_str(hash, MD5_DIGEST_LENGTH);

    // free memory
    free(a.elements);
    free(b.elements);
    free(c.elements);

    printf("{ \"valid\": true, \"n\": %d, \"kernel_time\": %.9f, \"cache_config\": \"none\", \"kernel\": \"mat_mult_seq\", \"hash\": \"%s\" size_int: %d}\n", n, seq_time, hash_str, sizeof(int));
    free(hash_str);
    
    return EXIT_SUCCESS;
}

void mat_mult_init(int argc, char** argv, int *n)
{
	assert(argc == 2);

	*n = atoi(argv[1]);

	assert(*n > 0) ;
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

