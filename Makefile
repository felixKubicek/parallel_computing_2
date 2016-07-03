
CC=/usr/local/apps/cuda/7.5/bin/nvcc
CFLAGS=-DVALIDATE


mat_mult: mat_mult.cu
	$(CC) -o $@  $< $(CFLAGS)
