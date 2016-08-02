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
static const cell_state_t anneal[10] = {0, 0, 0, 0, 1, 0, 1, 1, 1, 1};

/* treat torus like boundary conditions */
static void boundary(line_t *buf, int lines)
{
	for (int y = 0; y <= lines + 1; y++) {
		/* copy rightmost column to the buffer column 0 */
		buf[y][0] = buf[y][XSIZE];

		/* copy leftmost column to the buffer column XSIZE + 1 */
		buf[y][XSIZE + 1] = buf[y][1];
	}

	for (int x = 0; x <= XSIZE + 1; x++) {
		/* copy bottommost row to buffer row 0 */
		buf[0][x] = buf[lines][x];

		/* copy topmost row to buffer row lines + 1 */
		buf[lines + 1][x] = buf[1][x];
	}
}

/* make one simulation iteration with lines lines.
 * old configuration is in from, new one is written to to.
 */
void simulate(line_t_cuda *from, line_t_cuda *to, int lines)
{
        for (int y = 0; y < lines; y++) {
		for (int x = 0; x < XSIZE; x++) {

                        int x_prev = ((x - 1) + XSIZE) % XSIZE;
                        int y_prev = ((y - 1) + lines) % lines;
                        int x_next = (x + 1) % XSIZE; 
                        int y_next = (y + 1) % lines; 

			to[y][x] = transition_cuda(from, x_prev, y_prev, x, y, x_next, y_next);
		}
	}

        /*
        int x = (threadIdx.x + blockIdx.x * blockDim.x) + 1;
        int y = (threadIdx.y + blockIdx.y * blockDim.y) + 1;
        
        int x_prev = ((x - 1) + XSIZE) % XSIZE;
        int y_prev = ((y - 1) + lines) % lines;
        int x_next = (x + 1) % XSIZE; 
        int y_next = (y + 1) % lines; 

        to[y][x] = transition_torus(from, x_prev, y_prev, x, y, x_next, y_next);
        */

        // TODO: for loops for x and y dimension 
}

/* --------------------- measurement ---------------------------------- */

int main(int argc, char** argv)
{
	int lines, its;

	ca_init(argc, argv, &lines, &its);

        line_t_cuda *from, *to, *from_d, *to_d;

	from = (line_t_cuda *) calloc(lines, sizeof(line_t_cuda));
	to = (line_t_cuda *) calloc(lines, sizeof(line_t_cuda));

	ca_init_config_cuda(from, lines, 0);

	TIME_GET(sim_start);
	for (int i = 0; i < its; i++) {
		simulate(from, to, lines);

		line_t_cuda *temp = from;
		from = to;
		to = temp;
	}
	TIME_GET(sim_stop);

	line_t *verify_field = (line_t *) calloc((lines + 2), sizeof(line_t));

        for(int y = 1; y <= lines; y++)
        {
            memcpy((void *) &verify_field[y][1], (void *) &from[y-1][0], XSIZE);
        }
        
	ca_hash_and_report(verify_field + 1, lines, TIME_DIFF(sim_start, sim_stop));

	free(from);
	free(to);

	return EXIT_SUCCESS;
}
