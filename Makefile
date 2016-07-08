CC=/usr/local/apps/cuda/7.5/bin/nvcc
CFLAGS=-DVALIDATE
TARGET=mat_mult
N=64

$(TARGET)_$(N)_$(N): $(TARGET).cu
	$(CC) -o $@  $< $(CFLAGS) -DN=$(N)
clean:
	rm $(TARGET)_*
