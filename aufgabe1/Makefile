SHELL=/bin/bash
CC=gcc
CUDA_CC=/usr/local/apps/cuda/7.5/bin/nvcc

COMMON_CFLAGS=-O2
COMMON_LDFLAGS=-lcrypto -lrt

C99=-std=gnu99
CUDA_CFLAGS=-Xcompiler=$(C99)

C_DEPS=../aufgabe2/ca_common.c ../aufgabe2/random.c

CUDA_TARGETS=mat_mult mat_mult_tiling
TARGETS=mat_mult_seq $(CUDA_TARGETS) 

.PHONY: all
all: $(TARGETS)

mat_mult: mat_mult.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $(VALIDATE_DEF) $^ $(COMMON_LDFLAGS) -o $@

mat_mult_tiling: mat_mult.cu $(C_DEPS) 
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $(VALIDATE_DEF) -DTILING $^ $(COMMON_LDFLAGS) -o $@ 

mat_mult_seq: mat_mult_seq.c $(C_DEPS)
	$(CC) $(C99) $(COMMON_CFLAGS) $^ $(COMMON_LDFLAGS) -o $@

.PHONY: bench
PASSES=1
DIMS=1024 5120 10240 15360 20480 25600 31584 # 31.584^2 per matrix is the absolute maximum

bench: bench_cuda bench_seq
bench_cuda: $(CUDA_TARGETS)
	@for N in ${DIMS}; do \
		for CACHE_CONFIG in 1 2 3; do \
			for BINARY in $^; do \
				for i in {1..${PASSES}}; do \
					./$$BINARY $$N $$CACHE_CONFIG; \
				done; \
			done; \
		done; \
	done;

bench_seq: mat_mult_seq
	@for N in ${DIMS}; do \
		for i in {1..${PASSES}}; do \
			./$< $$N; \
		done; \
	done;

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(TARGETS)
