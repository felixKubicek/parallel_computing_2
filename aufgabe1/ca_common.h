#ifndef CA_COMMON_H
#define CA_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <time.h>

/* a: pointer to array; x,y: coordinates; result: n-th element of anneal,
      where n is the number of neighbors */
#define transition(a, x, y) \
   (anneal[(a)[(y)-1][(x)-1] + (a)[(y)][(x)-1] + (a)[(y)+1][(x)-1] +\
           (a)[(y)-1][(x)  ] + (a)[(y)][(x)  ] + (a)[(y)+1][(x)  ] +\
           (a)[(y)-1][(x)+1] + (a)[(y)][(x)+1] + (a)[(y)+1][(x)+1]])

#define TIME_GET(timer) \
	struct timespec timer; \
	clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2) \
	((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) - \
	 (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) / 1.0E+9


#ifdef __cplusplus
extern "C" {
#endif

/* horizontal size of the configuration */
#define XSIZE 1024
#define LINE_SIZE (XSIZE + 2)

/* "ADT" State and line of states (plus border) */
typedef uint8_t cell_state_t;
typedef cell_state_t line_t[XSIZE + 2];

void ca_init(int argc, char** argv, int *lines, int *its);
void ca_init_config(line_t *buf, int lines, int skip_lines);
void ca_hash_and_report(line_t *buf, int lines, double time_in_s);

#ifdef __cplusplus
}
#endif

#ifdef USE_MPI

/* next/prev process in communicator */
#define PREV_PROC(n, num_procs) ((n - 1 + num_procs) % num_procs)
#define SUCC_PROC(n, num_procs) ((n + 1) % num_procs)

#define CA_MPI_CELL_DATATYPE MPI_BYTE

void ca_mpi_init(int num_procs, int rank, int num_total_lines,
		int *num_local_lines, int *global_first_line);
void ca_mpi_hash_and_report(line_t* local_buf, int num_local_lines,
		int num_total_lines, int num_procs, double time_in_s);

#endif /* USE_MPI */


#endif /* CA_COMMON_H */
