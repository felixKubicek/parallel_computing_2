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

/* --------------------- CA simulation -------------------------------- */

/* annealing rule from ChoDro96 page 34
 * the table is used to map the number of nonzero
 * states in the neighborhood to the new state
 */
__constant__ static const cell_state_t anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};

/* make one simulation iteration with lines lines.
 * old configuration is in from, new one is written to to.
 */
__global__ static void simulate(line_t_bit *from, line_t_bit *to, int lines)
{
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        int grid_size = blockDim.x * gridDim.x;
        
        int field_width = XSIZE/8;
        int field_size = field_width * lines;

        for (int i = gid; i < field_size; i+= grid_size)
        {
            int x = i % field_width;
            int y = i / field_width;

            int x_prev = ((x - 1) + field_width) % field_width;
            int y_prev = ((y - 1) + lines) % lines;
            int x_next = (x + 1) % field_width;
            int y_next = (y + 1) % lines;

            cell_state_t top_l = from[y_prev][x_prev] >> 7;
            cell_state_t top_r = from[y_prev][x_next] & 1;
            int top = (top_r << 9) + (from[y_prev][x] << 1) + top_l;

            cell_state_t middle_l = from[y][x_prev] >> 7;
            cell_state_t middle_r =  from[y][x_next] & 1;
            int  middle =  (middle_r << 9) + (from[y][x] << 1) + middle_l;

            cell_state_t bottom_l = from[y_next][x_prev] >> 7;
            cell_state_t bottom_r = from[y_next][x_next] & 1;
            int bottom =  (bottom_r << 9) + (from[y_next][x] << 1) + bottom_l;

            cell_state_t result = 0;
            int mask = 7;

            for(int b = 0; b < 8; b++)
            {
                int num_bits = ((top & mask) << 6) + ((middle & mask) << 3) + (bottom & mask);
                num_bits = (num_bits & 0x5555) + ((num_bits >> 1) & 0x5555);
                num_bits = (num_bits & 0x3333) + ((num_bits >> 2) & 0x3333);
                num_bits = (num_bits & 0x0f0f) + ((num_bits >> 4) & 0x0f0f);
                num_bits = (num_bits & 0x00ff) + ((num_bits >> 8) & 0x00ff);

                result |= (anneal[num_bits] << b);
                mask = mask << 1;
            }

            to[y][x] = result;
        }
}

/* --------------------- measurement ---------------------------------- */

int main(int argc, char** argv)
{
	int lines, its;

	ca_init(argc, argv, &lines, &its);

        line_t_bit *from, *to, *from_d, *to_d;
        line_t *verify_field;

	MALLOC_ERROR_CHECK(from = (line_t_bit *) calloc(lines, sizeof(line_t_bit)));
	MALLOC_ERROR_CHECK(to = (line_t_bit *) calloc(lines, sizeof(line_t_bit)));
	MALLOC_ERROR_CHECK(verify_field = (line_t *) calloc((lines + 2), sizeof(line_t)));

        CUDA_ERROR_CHECK(cudaMalloc((void **) &from_d, lines * sizeof(line_t_bit)));
        CUDA_ERROR_CHECK(cudaMalloc((void **) &to_d, lines * sizeof(line_t_bit)));
        CUDA_ERROR_CHECK(cudaMalloc((void **) &to_d, lines * sizeof(line_t_bit)));

	ca_init_config_bit(from, lines, 0);

        CUDA_ERROR_CHECK(cudaMemcpy((void *) from_d, (void *) from, lines * sizeof(line_t_bit), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy((void *) to_d, (void *) to, lines * sizeof(line_t_bit), cudaMemcpyHostToDevice));

	TIME_GET(sim_start);
	for (int i = 0; i < its; i++) 
        {
                simulate <<<lines, XSIZE/8>>> (from_d, to_d, lines);

                line_t_bit *temp = from_d;
		from_d = to_d;
		to_d = temp;
	}
        cudaDeviceSynchronize();
	TIME_GET(sim_stop);

        CUDA_ERROR_CHECK(cudaPeekAtLastError());
        CUDA_ERROR_CHECK(cudaMemcpy((void *) from, (void *) from_d, lines * sizeof(line_t_bit), cudaMemcpyDeviceToHost));

        for (int y = 0; y < lines; y++)
        {
            for (int x = 0; x < XSIZE; x++)
            {
                 verify_field[y+1][x+1] = get_bit(from, x, y);
            }
        }

	ca_hash_and_report(verify_field + 1, lines, TIME_DIFF(sim_start, sim_stop));
        
        free(from);
	free(to);
        free(verify_field);
        CUDA_ERROR_CHECK(cudaFree(from_d));
        CUDA_ERROR_CHECK(cudaFree(to_d));

	return EXIT_SUCCESS;
}
