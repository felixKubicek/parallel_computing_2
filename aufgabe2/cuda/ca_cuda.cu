/*
 * simulate a cellular automaton with periodic boundaries (torus-like)
 * serial version
 *
 * (c) 2016 Steffen Christgau (C99 port, modularization)
 * (c) 1996,1997 Peter Sanders, Ingo Boesnach (original source)
 *
 * command line arguments:
 * #1: Number of lines
 * #2: Number of iterations to be simulated
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "ca_common.h"

#ifdef USE_2D_MAPPING  
    #define BLOCK_SIZE_X 32
    #define BLOCK_SIZE_Y 4
#endif

#define CUDA_ERROR_CHECK(x)\
    do {cudaError_t last_err = (x);\
        if (last_err != cudaSuccess)\
                {fprintf(stderr ,"%s:%u: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString ( last_err )); exit(EXIT_FAILURE); }\
    } while (false)

#define MALLOC_ERROR_CHECK(x)\
        do {\
            if ( (x) == NULL)\
                {fprintf(stderr ,"%s:%u: malloc error!\n", __FILE__, __LINE__); exit (EXIT_FAILURE); }\
        } while (false)

/* --------------------- CA simulation -------------------------------- */

/* annealing rule from ChoDro96 page 34
 * the table is used to map the number of nonzero
 * states in the neighborhood to the new state
 */
__constant__ static const cell_state_t anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};

/* make one simulation iteration with lines lines.
 * old configuration is in from, new one is written to to.
 */
__global__ static void simulate(line_t_cuda *from, line_t_cuda *to, int lines)
{
#ifdef USE_2D_MAPPING  
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        
        int x_prev = ((x - 1) + XSIZE) % XSIZE;
        int y_prev = ((y - 1) + lines) % lines;
        int x_next = (x + 1) % XSIZE; 
        int y_next = (y + 1) % lines;
                    
        to[y][x] = transition_cuda(from, x_prev, y_prev, x, y, x_next, y_next);
#else
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        int grid_size = blockDim.x * gridDim.x;

        for (int i = gid; i < lines * XSIZE; i+= grid_size)
        {
            int x = i % XSIZE;
            int y = i / XSIZE;
            
            int x_prev = ((x - 1) + XSIZE) % XSIZE;
            int y_prev = ((y - 1) + lines) % lines;
            int x_next = (x + 1) % XSIZE; 
            int y_next = (y + 1) % lines;
                    
            to[y][x] = transition_cuda(from, x_prev, y_prev, x, y, x_next, y_next);
        }
#endif
}

/* --------------------- measurement ---------------------------------- */

int main(int argc, char** argv)
{
	int lines, its;

	ca_init(argc, argv, &lines, &its);

        line_t_cuda *from, *to, *from_d, *to_d;
        line_t *verify_field;

	MALLOC_ERROR_CHECK(from = (line_t_cuda *) calloc(lines, sizeof(line_t_cuda)));
	MALLOC_ERROR_CHECK(to = (line_t_cuda *) calloc(lines, sizeof(line_t_cuda)));
	MALLOC_ERROR_CHECK(verify_field = (line_t *) calloc((lines + 2), sizeof(line_t)));

        CUDA_ERROR_CHECK(cudaMalloc((void **) &from_d, lines * sizeof(line_t_cuda)));
        CUDA_ERROR_CHECK(cudaMalloc((void **) &to_d, lines * sizeof(line_t_cuda)));
        CUDA_ERROR_CHECK(cudaMalloc((void **) &to_d, lines * sizeof(line_t_cuda)));

	ca_init_config_cuda(from, lines, 0);
        CUDA_ERROR_CHECK(cudaMemcpy((void *) from_d, (void *) from, lines * sizeof(line_t_cuda), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy((void *) to_d, (void *) to, lines * sizeof(line_t_cuda), cudaMemcpyHostToDevice));

#ifdef USE_2D_MAPPING  
        dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        dim3 dimGrid(XSIZE/dimBlock.x, lines/dimBlock.y);
#endif
	TIME_GET(sim_start);
	for (int i = 0; i < its; i++) 
        {
#ifdef USE_2D_MAPPING  
		simulate <<<dimGrid, dimBlock>>> (from_d, to_d, lines);
#else
                simulate <<<lines, XSIZE/4>>> (from_d, to_d, lines);
#endif
                line_t_cuda *temp = from_d;
		from_d = to_d;
		to_d = temp;
	}
        cudaDeviceSynchronize();
	TIME_GET(sim_stop);

        CUDA_ERROR_CHECK(cudaPeekAtLastError());
        CUDA_ERROR_CHECK(cudaMemcpy((void *) from, (void *) from_d, lines * sizeof(line_t_cuda), cudaMemcpyDeviceToHost));

        for(int y = 1; y <= lines; y++)
        {
            memcpy((void *) &verify_field[y][1], (void *) &from[y-1][0], XSIZE);
        }

	ca_hash_and_report(verify_field + 1, lines, TIME_DIFF(sim_start, sim_stop));

	free(from);
	free(to);
        free(verify_field);
        CUDA_ERROR_CHECK(cudaFree(from_d));
        CUDA_ERROR_CHECK(cudaFree(to_d));

	return EXIT_SUCCESS;
}
