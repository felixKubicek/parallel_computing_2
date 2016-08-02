/*
 * common functions for (parallel) cellular automaton
 *
 * (c) 2016 Steffen Christgau
 *
 * configuration initialization based on
 * (c) 1996,1997 Peter Sanders, Ingo Boesnach
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "openssl/md5.h"

#include "ca_common.h"
#include "random.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

/* determine random integer between 0 and n-1 */
#define randInt(n) ((int)(nextRandomLEcuyer() * n))

void print_field(line_t *field, int lines)
{
    for(int y = 0; y < lines; y++)
    {
        for (int x = 0; x < XSIZE+2; x++)
        {
            printf("%d", field[y][x]);   
        }
        printf("\n");
    }
}

void ca_init(int argc, char** argv, int *lines, int *its)
{
	assert(argc == 3);

	*lines = atoi(argv[1]);
	*its = atoi(argv[2]);

	assert(*lines > 0);
}

/* random starting configuration */
void ca_init_config(line_t *buf, int lines, int skip_lines)
{
	volatile int scratch;

	initRandomLEcuyer(424243);

	/* let the RNG spin for some rounds (used for distributed initialization) */
	for (int y = 1;  y <= skip_lines;  y++) {
		for (int x = 1;  x <= XSIZE;  x++) {
			scratch = scratch + randInt(100) >= 50;
		}
	}

	for (int y = 1;  y <= lines;  y++) {
		for (int x = 1;  x <= XSIZE;  x++) {
			buf[y][x] = randInt(100) >= 50;
		}
	}
}

/* random starting configuration for cuda (without ghostzones) */
void ca_init_config_cuda(line_t *buf, int lines, int skip_lines)
{
	volatile int scratch;

	initRandomLEcuyer(424243);

	/* let the RNG spin for some rounds (used for distributed initialization) */
	for (int y = 0;  y < skip_lines;  y++) {
		for (int x = 1;  x <= XSIZE;  x++) {
			scratch = scratch + randInt(100) >= 50;
		}
	}

	for (int y = 0;  y < lines;  y++) {
		for (int x = 1;  x <= XSIZE;  x++) {
			buf[y][x] = randInt(100) >= 50;
		}
	}
}

static char* ca_buffer_to_hex_str(const uint8_t* buf, size_t buf_size)
{
  char *retval, *ptr;

  retval = ptr = calloc(MD5_DIGEST_LENGTH * 2 + 1, sizeof(*retval));
  for (size_t i = 0; i < MD5_DIGEST_LENGTH; i++) {
    snprintf(ptr, 3, "%02X", buf[i]);
    ptr += 2;
  }

  return retval;
}

static void ca_print_hash_and_time(const char const *hash, const double time)
{
	printf("hash: %s\ttime: %.3f s\n", (hash ? hash : "ERROR"), time);
}

static void ca_clean_ghost_zones(line_t *buf, int lines)
{
	for (int y = 0; y < lines; y++) {
		buf[y][0] = 0;
		buf[y][XSIZE + 1] = 0;
	}
}

void ca_hash_and_report(line_t *buf, int lines, double time_in_s)
{
	uint8_t hash[MD5_DIGEST_LENGTH];
	MD5_CTX ctx;

	ca_clean_ghost_zones(buf, lines);

	MD5_Init(&ctx);
	MD5_Update(&ctx, buf, lines * sizeof(*buf));
	MD5_Final(hash, &ctx);

	char* hash_str = ca_buffer_to_hex_str(hash, MD5_DIGEST_LENGTH);
	ca_print_hash_and_time(hash_str, time_in_s);
	free(hash_str);
}

#ifdef MPI_VERSION /* defined by mpi.h */

static int num_remainder_procs;
#ifdef USE_MPI_TOPOLOGY
static MPI_Comm topo_comm;
#endif

void ca_mpi_init(int num_procs, int rank, int num_total_lines,
		int *num_local_lines, int *global_first_line)
{
	*num_local_lines = num_total_lines / num_procs;
	*global_first_line = rank * (*num_local_lines);


	/* if work cannot be distributed equally, distribute the remaining lines equally */
	num_remainder_procs = num_total_lines % num_procs;
	if (rank < num_remainder_procs) {
		(*num_local_lines)++;
		*global_first_line = *global_first_line + rank;
	} else {
		*global_first_line = *global_first_line + num_remainder_procs;
	}

#ifdef USE_TOPO
	int topo_periodic = 1, topo_dim = num_procs;
	MPI_Cart_create(MPI_COMM_WORLD, 1, &topo_dim, &topo_periodic, 0, &topo_comm);
#endif
}

#define TAG_RESULT (0xCAFE)

void ca_mpi_hash_and_report(line_t* local_buf, int num_local_lines,
		int num_total_lines, int num_procs, double time_in_s)
{
	MD5_CTX ctx;
	int i, rank, num_lines = num_local_lines, count;
	uint8_t hash[MD5_DIGEST_LENGTH];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {

		MD5_Init(&ctx);
		count = num_local_lines;
		ca_clean_ghost_zones(local_buf + 1, num_local_lines);
	    /* insert our own data into MD5 hash */
		MD5_Update(&ctx, local_buf + 1, num_local_lines * sizeof(line_t));

	    /* recieve partial results from all other processes in our local buffer and
		 * update the hash. Our buffer is garanteed to have the maximum required
	     * size in any case (see partioning above) */
		for (i = 1; i < num_procs; i++) {
			num_lines = num_total_lines / num_procs;
			if (i < num_remainder_procs) {
				num_lines++;
			}
			count += num_lines;
			MPI_Recv(
				local_buf, num_lines * LINE_SIZE, CA_MPI_CELL_DATATYPE,
				i, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			ca_clean_ghost_zones(local_buf, num_lines);
			MD5_Update(&ctx, local_buf, num_lines * sizeof(line_t));
		}

		MD5_Final(hash, &ctx);

		char* hash_str = ca_buffer_to_hex_str(hash, MD5_DIGEST_LENGTH);
		ca_print_hash_and_time(hash_str, time_in_s);

		free(hash_str);
	} else {
		MPI_Send(
			local_buf[1], num_local_lines * LINE_SIZE, CA_MPI_CELL_DATATYPE,
			0, TAG_RESULT, MPI_COMM_WORLD);
	}

#ifdef USE_MPI_TOPOLOGY
	MPI_Comm_free(&topo_comm);
#endif
}

#endif /* MPI_VERSION */
