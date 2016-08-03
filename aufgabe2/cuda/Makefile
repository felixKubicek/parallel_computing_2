BASE_CC=gcc
ACC_CC=pgcc
CUDA_CC=/usr/local/apps/cuda/7.5/bin/nvcc
MPI_CC=mpicc

COMMON_CFLAGS=-O2
COMMON_LDFLAGS=-lcrypto -lrt

BASE_CFLAGS=-Wall -std=gnu99 -pedantic
BASE_OMP_CFLAGS=$(BASE_CFLAGS) -fopenmp

ACC_CFLAGS=-acc -Minfo=accel -noswitcherror -Mlarge_arrays #-ta=tesla:cc20,cc35

CUDA_CFLAGS=-Xcompiler=-std=gnu99 \
			#-lineinfo \
			-gencode=arch=compute_20,code=sm_20 \
			-gencode=arch=compute_20,code=compute_20 \
			-gencode=arch=compute_35,code=sm_35 \
			-gencode=arch=compute_35,code=compute_35 \

MPI_CFLAGS=-DUSE_MPI

C_DEPS=ca_common.c random.c

MPI_TARGETS=ca_mpi_p2p ca_mpi_p2p_nb ca_mpi_osc

TARGETS=ca_cuda_2d
        #ca_seq \
	#ca_openmp \
	#ca_cuda_1d ca_openacc \
	#$(MPI_TARGETS)

.PHONY: all
all: $(TARGETS)

.PHONY: gpu
gpu: ca_cuda ca_openacc

.PHONY: cpu
cpu: ca_seq ca_openmp

.PHONY: mpi
mpi: $(MPI_TARGETS)

ca_seq: ca_seq.c $(C_DEPS)
	$(BASE_CC) $(COMMON_CFLAGS) $(BASE_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_openmp: ca_openmp.c $(C_DEPS)
	$(BASE_CC) $(COMMON_CFLAGS) $(BASE_OMP_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_cuda_1d: ca_cuda.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_cuda_2d: ca_cuda.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) -DUSE_2D_MAPPING $^ $(COMMON_LDFLAGS) -o $@

ca_openacc: ca_openacc.c $(C_DEPS)
	$(ACC_CC) $(COMMON_CFLAGS) $(ACC_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

ca_mpi_p2p: ca_mpi_p2p.c $(C_DEPS)
	$(MPI_CC) $(COMMON_CFLAGS) $(BASE_CFLAGS) $(MPI_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@
	
ca_mpi_p2p_nb: ca_mpi_p2p_nb.c $(C_DEPS)
	$(MPI_CC) $(COMMON_CFLAGS) $(BASE_CFLAGS) $(MPI_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@
   
ca_mpi_osc: ca_mpi_osc.c $(C_DEPS)
	$(MPI_CC) $(COMMON_CFLAGS) $(BASE_CFLAGS) $(MPI_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

.PHONY: test

test: $(TARGETS)
	@for ITS in 10 31 57 100; do \
		for LINES in 10 33 47 100; do \
			echo "$$LINES lines, $$ITS iterations"; \
			for BINARY in $^; do printf '%-10s\t' $$BINARY; ./$$BINARY $$LINES $$ITS; done; \
		done \
	done

.PHONY: mpi-test

mpi-test: $(MPI_TARGETS)
	@for ITS in 10 31 57 100; do \
		for LINES in 20 33 47 100; do \
			for NP in 2 3 4; do \
				echo "$$LINES lines, $$ITS iterations, $$NP procs"; \
				for BINARY in ca_seq $^; do \
					printf '%-10s\t' $$BINARY; \
					mpiexec -n $$NP ./$$BINARY $$LINES $$ITS; \
				done \
			done \
		done \
	done

.PHONY: bench

bench: $(TARGETS)
	@for ITS in 128 256 512; do \
		for LINES in 1000 10000 50000; do \
			echo "$$LINES lines, $$ITS iterations"; \
			for BINARY in $^; do printf '%-10s\t' $$BINARY; ./$$BINARY $$LINES $$ITS; done; \
		done \
	done

.PHONY: clean

clean:
	rm -f *.o
	rm -f $(TARGETS)
