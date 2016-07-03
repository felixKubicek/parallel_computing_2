CC=/usr/local/apps/cuda/7.5/bin/nvcc
CFLAGS=-DVALIDATE
TARGET=mat_mult

$(TARGET): $(TARGET).cu
	$(CC) -o $@  $< $(CFLAGS)
clean:
	rm $(TARGET)
