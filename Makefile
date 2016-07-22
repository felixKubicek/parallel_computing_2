CC=/usr/local/apps/cuda/7.5/bin/nvcc
CFLAGS=-DVALIDATE
TARGET=mat_mult
N=64
CACHE_CONFIG=0

$(TARGET)_$(N)_$(CACHE_CONFIG): $(TARGET).cu
	$(CC) -o $@  $< $(CFLAGS) -DN=$(N) -DCUDA_CACHE_CONFIG=$(CACHE_CONFIG)
clean:
	rm $(TARGET)_*
