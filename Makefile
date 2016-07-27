CC=/usr/local/apps/cuda/7.5/bin/nvcc
XCFLAGS= # emable tiling with -DTILING
TARGET=mat_mult
N=64
CUDA_CACHE_CONFIG=0 # == cudaFuncCachePreferNone

$(TARGET): $(TARGET).cu
	$(CC) -o $@  $< $(XCFLAGS) -DN=$(N) -DCUDA_CACHE_CONFIG=$(CUDA_CACHE_CONFIG)
clean:
	rm $(TARGET)
