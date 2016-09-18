SHELL=/bin/bash
BASE_CC=gcc
ACC_CC=/usr/local/apps/openacc/15.10/linux86-64/15.10/bin/pgcc
CUDA_CC=/usr/local/apps/cuda/7.5/bin/nvcc
MPI_CC=mpicc

COMMON_CFLAGS=-O2
COMMON_LDFLAGS=-lcrypto -lrt

BASE_CFLAGS=-Wall -std=gnu99 -pedantic
BASE_OMP_CFLAGS=$(BASE_CFLAGS) -fopenmp

ACC_CFLAGS=-acc -Minfo=accel -noswitcherror -Mlarge_arrays #-ta=nvidia,time

CUDA_CFLAGS=-Xcompiler=-std=gnu99 -gencode=arch=compute_35,code=sm_35 #-lineinfo 

CL_CFLAGS=-lOpenCL

C_DEPS=ca_common.c random.c

MPI_TARGETS=ca_mpi_p2p ca_mpi_p2p_nb ca_mpi_osc

TARGETS=ca_cuda_2d #ca_seq ca_cuda_1d  ca_opencl ca_openacc 

.PHONY: all
all: $(TARGETS)

.PHONY: gpu
gpu: ca_cuda ca_openacc

.PHONY: cpu
cpu: ca_seq ca_openmp


ca_seq: ca_seq.c $(C_DEPS)
	$(BASE_CC) $(COMMON_CFLAGS) $(BASE_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_openmp: ca_openmp.c $(C_DEPS)
	$(BASE_CC) $(COMMON_CFLAGS) $(BASE_OMP_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_cuda_1d: ca_cuda.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_cuda_2d: ca_cuda.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) -DUSE_2D_MAPPING $^ $(COMMON_LDFLAGS) -o $@

ca_cuda_bit: ca_cuda_bit.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_opencl: ca_opencl.c $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $(CL_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@ 

ca_openacc: ca_openacc.c $(C_DEPS)
	$(ACC_CC) $(COMMON_CFLAGS) $(ACC_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@


.PHONY: test

test: $(TARGETS)
	@for ITS in 10 31 57 100; do \
		for LINES in 10 33 47 100; do \
			echo "$$LINES lines, $$ITS iterations"; \
			for BINARY in $^; do printf '%-10s\t' $$BINARY; ./$$BINARY $$LINES $$ITS; done; \
		done \
	done

.PHONY: bench

PASSES=1
bench: $(TARGETS)
	@for ITS in 1000 2000 3000; do \
		for LINES in 1000 10000 100000; do \
			echo "$$LINES lines, $$ITS iterations"; \
			for BINARY in $^; do \
				for i in {1..${PASSES}}; do \
					printf '%-10s\t' $$BINARY; ./$$BINARY $$LINES $$ITS; \
				done; \
			done; \
		done; \
	done;

.PHONY: clean

clean:
	rm -f *.o
	rm -f $(TARGETS)
