#include "pnet.hh"
#include "hazard.hh"
#include <boost/numeric/ublas/matrix.hpp>
#include <cuda.h>

using namespace boost::numeric::ublas;

typedef double (*rl_pointer)(double, int *, int *);

__global__ void AddVecKernel(float * i, float * j, float * p, int width);

void Pnet::AddVec(float * i, float * j, float * p, int width) {
	int size = width * sizeof(float);
	float * id, * jd, * pd;

	cudaMalloc((void**) &id, size);
	cudaMemcpy(id, i, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &jd, size);
	cudaMemcpy(jd, j, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &pd, size);

	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(width/1024 + 1, 1, 1);

	AddVecKernel<<<dimGrid, dimBlock>>>(id, jd, pd, width);

	cudaMemcpy(p, pd, size, cudaMemcpyDeviceToHost);
	cudaFree(id); cudaFree(jd); cudaFree(pd);
}

__global__ void AddVecKernel(float * id, float * jd, float * pd, int width) {
	int tx = threadIdx.x;

	pd[tx] = id[tx] + jd[tx];
}
