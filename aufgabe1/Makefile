SHELL=/bin/bash
CUDA_CC=/usr/local/apps/cuda/7.5/bin/nvcc

TARGETS=mat_mult mat_mult_tiling

.PHONY: all
all: $(TARGETS)

mat_mult: mat_mult.cu
	$(CUDA_CC) -o $@  $< 

mat_mult_tiling: mat_mult.cu
	$(CUDA_CC) -o $@  $< -DTILING 

.PHONY: bench
PASSES=1
bench: $(TARGETS)
	@for N in 3200; do \
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
