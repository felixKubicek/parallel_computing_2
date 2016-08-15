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

#define transition_cuda(a, x_prev, y_prev, x, y, x_next, y_next) \
   (anneal[(a)[y_prev][x_prev] + (a)[(y)][x_prev] + (a)[y_next][x_prev] +\
           (a)[y_prev][(x)   ] + (a)[(y)][(x)   ] + (a)[y_next][(x)   ] +\
           (a)[y_prev][x_next] + (a)[(y)][x_next] + (a)[y_next][x_next]])

#define TIME_GET(timer) \
	struct timespec timer; \
	clock_gettime(CLOCK_MONOTONIC, &timer)

#define TIME_DIFF(timer1, timer2) \
	((timer2.tv_sec * 1.0E+9 + timer2.tv_nsec) - \
	 (timer1.tv_sec * 1.0E+9 + timer1.tv_nsec)) / 1.0E+9

#define MALLOC_ERROR_CHECK(x)\
        do {\
            if ( (x) == NULL)\
                {fprintf(stderr ,"%s:%u: malloc error!\n", __FILE__, __LINE__); exit (EXIT_FAILURE); }\
        } while (0)

#define CUDA_ERROR_CHECK(x)\
    do {cudaError_t last_err = (x);\
        if (last_err != cudaSuccess)\
                {fprintf(stderr ,"%s:%u: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString ( last_err )); exit(EXIT_FAILURE); }\
    } while (false)

#define STR(s) XSTR(s)
#define XSTR(s) #s

#ifdef __cplusplus
extern "C" {
#endif

/* horizontal size of the configuration */
#define XSIZE 1024
#define LINE_SIZE (XSIZE + 2)

/* "ADT" State and line of states (plus border) */
typedef uint8_t cell_state_t;
typedef cell_state_t line_t[XSIZE + 2];
typedef cell_state_t line_t_cuda[XSIZE];

void ca_init(int argc, char** argv, int *lines, int *its);
void ca_init_config(line_t *buf, int lines, int skip_lines);
void ca_init_config_cuda(line_t_cuda *buf, int lines, int skip_lines);
void ca_hash_and_report(line_t *buf, int lines, double time_in_s);
void print_field(line_t *field, int lines);
void print_field_cuda(line_t_cuda *field, int lines);

#ifdef __cplusplus
}
#endif

#endif /* CA_COMMON_H */

