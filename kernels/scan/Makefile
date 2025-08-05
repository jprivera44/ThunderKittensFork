# GPU Selection: H100
GPU_TARGET=H100

# Compiler
NVCC?=nvcc
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler=-fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -std=c++20 -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype

# Targets
TARGETS=scan test_scan

# Default target
all: $(TARGETS)

scan: pc.cu
	$(NVCC) pc.cu $(NVCCFLAGS) -o scan

test_scan: test_scan.cu
	$(NVCC) test_scan.cu $(NVCCFLAGS) -o test_scan

# Clean target
clean:
	rm -f $(TARGETS)

# Run test
test: test_scan
	./test_scan