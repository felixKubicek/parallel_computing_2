SHELL=/bin/bash
CUDA_CC=/usr/local/apps/cuda/7.5/bin/nvcc

COMMON_CFLAGS=-O2
COMMON_LDFLAGS=-lcrypto -lrt

CUDA_CFLAGS=-Xcompiler=-std=gnu99

C_DEPS=../aufgabe2/ca_common.c ../aufgabe2/random.c

TARGETS=mat_mult mat_mult_tiling

.PHONY: all
all: $(TARGETS)

mat_mult: mat_mult.cu $(C_DEPS)
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $(VALIDATE_DEF) $^ $(COMMON_LDFLAGS) -o $@

mat_mult_tiling: mat_mult.cu $(C_DEPS) 
	$(CUDA_CC) $(COMMON_CFLAGS) $(CUDA_CFLAGS) $(VALIDATE_DEF) -DTILING $^ $(COMMON_LDFLAGS) -o $@ 

.PHONY: bench
PASSES=1
bench: $(TARGETS)
	@for N in 32 64; do \
		for CACHE_CONFIG in 1; do \
			for BINARY in $^; do \
				for i in {1..${PASSES}}; do \
					./$$BINARY $$N $$CACHE_CONFIG; \
				done; \
			done; \
		done; \
	done;

.PHONY: clean
clean:
	rm -f *.o
	rm -f $(TARGETS)
