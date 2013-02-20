#define PROFILE 1
#define FLOATT float

#include <cuda.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "hazard.hh"

#if PROFILE
#include "Profile.h"
#include "ProfileCodes.h"
#endif

PROF_ALLOC;

using namespace boost::numeric::ublas;

typedef FLOATT (*rl_pointer)(FLOATT, int, int);

int * Mh;
__device__ int* Md;
__device__ FLOATT * cd;
__device__ rl_pointer * HFuncd;
__device__ FLOATT * Hd;

int sizemd;
int sizec;
int sizehfunc;
int sizeh;

__global__ void UpdateKernel(int * Md, const FLOATT * cd, const rl_pointer * HFuncd, FLOATT * Hd, const int width);
__device__ FLOATT rl_00(FLOATT c, int  M1, int  M2);
__device__ FLOATT rl_10(FLOATT c, int  M1, int  M2);
__device__ FLOATT rl_11(FLOATT c, int  M1, int  M2);
__device__ FLOATT rl_20(FLOATT c, int  M1, int  M2);

__device__ FLOATT rl_00(FLOATT c, int  M1, int  M2) {
    //std::cout << "rl_00" << std::endl;
	return c;
}

__device__ FLOATT rl_10(FLOATT c, int  M1, int  M2) {
	//std::cout << "rl_10" << '\t' << M1 << std::endl;
	return c*M1;
}

__device__ FLOATT rl_11(FLOATT c, int  M1, int  M2) {
    //std::cout << "rl_11" << '\t' << M1 << '\t' << M2 << std::endl;
	return c*M1*M2;
}

__device__ FLOATT rl_20(FLOATT c, int  M1, int  M2) {
    //std::cout << "rl_20" << '\t' << M1 << std::endl;
	return (c/2)*M1*(M1-1);
}

const __device__ rl_pointer prl_00 = rl_00;
const __device__ rl_pointer prl_10 = rl_10;
const __device__ rl_pointer prl_20 = rl_20;
const __device__ rl_pointer prl_11 = rl_11;

void Hazard::MallocGlobal() {
	sizemd = MPtrs.size1() * 2 * sizeof(int);
	sizec = c.size1() * sizeof(FLOATT);
	sizehfunc = HFunc.size1() * sizeof(rl_pointer);
	sizeh = H.size1() * sizeof(FLOATT);
	Mh = (int*)malloc(sizemd);
	cudaMalloc((void**) &Md, sizemd);
	cudaMalloc((void**) &cd, sizec);
	cudaMalloc((void**) &HFuncd, sizehfunc);
	cudaMalloc((void**) &Hd, sizeh);
}

matrix<rl_pointer> Hazard::InitHFunc() {
	matrix<rl_pointer> hfunc(c.size1(), 1);
	int max1;
	int max2;
	int max1i;
	int max2i;
	for (matrix<int>::const_iterator1 it1 = Pre.begin1(); it1!=Pre.end1(); ++it1) {
		max1 = 0;
		max2 = 0;
		max1i = -1;
		max2i = -1;
		for (matrix<int>::const_iterator2 it2 = it1.begin(); it2!=it1.end(); ++it2) {
			if (*it2 > max1) {
				max2 = max1;
				max2i = max1i;
				max1 = *it2;
				max1i = it2.index2();
			}
			else if (*it2 > max2) {
				max2 = *it2;
				max2i = it2.index2();
			}
		}
		if (max1==0 && max2==0)
			cudaMemcpyFromSymbol(&(hfunc.data()[0]) + it1.index1(), prl_00, sizeof(rl_pointer));
		else if (max1==1 && max2==0)
			cudaMemcpyFromSymbol(&(hfunc.data()[0]) + it1.index1(), prl_10, sizeof(rl_pointer));
		else if (max1==1 && max2==1)
			cudaMemcpyFromSymbol(&(hfunc.data()[0]) + it1.index1(), prl_11, sizeof(rl_pointer));
		else if (max1==2 && max2==0)
			cudaMemcpyFromSymbol(&(hfunc.data()[0]) + it1.index1(), prl_20, sizeof(rl_pointer));
		else
			throw;
	}
	return hfunc;
}

matrix<int> Hazard::InitMPtrs(matrix<int> &M) {
	matrix<int> mptrs(c.size1(), 2);
	int max1;
	int max2;
	int max1i;
	int max2i;
	for (matrix<int>::const_iterator1 it1 = Pre.begin1(); it1!=Pre.end1(); ++it1) {
		max1 = 0;
		max2 = 0;
		max1i = -1;
		max2i = -1;
		for (matrix<int>::const_iterator2 it2 = it1.begin(); it2!=it1.end(); ++it2) {
			if (*it2 > max1) {
				max2 = max1;
				max2i = max1i;
				max1 = *it2;
				max1i = it2.index2();
			}
			else if (*it2 > max2) {
			    //std::cout << " max2 assigned " << max2 << ' ' << *it2 << ' ' << it2.index2() << ' ';
				max2 = *it2;
				max2i = it2.index2();
			}
		}
		//std::cout << '\t' << max1 << '\t' << max2 << '\t' << max1i << '\t' << max2i << '\t';
		if (max1 > 0)
			mptrs(it1.index1(), 0) = max1i;
		else
			mptrs(it1.index1(), 0) = -1;
		if (max2 > 0)
			mptrs(it1.index1(), 1) = max2i;
		else
			mptrs(it1.index1(), 1) = -1;
	    //std::cout << "M address in initmptrs: " << &(M.data()[0]) << std::endl;
	}
	return mptrs;
}

void Hazard::Update(matrix<int> M) {
	static const scalar_matrix<FLOATT> summer (scalar_matrix<FLOATT> (1, H.size1(), 1));

//	int * Mh, * Md;
//	FLOATT * cd;
//	rl_pointer * HFuncd;
//	FLOATT * Hd;
//
//	const int sizemd = MPtrs.size1() * 2 * sizeof(int);
//	const int sizec = c.size1() * sizeof(FLOATT);
//	const int sizehfunc = HFunc.size1() * sizeof(rl_pointer);
//	const int sizeh = H.size1() * sizeof(FLOATT);

	//std::cout << sizemd << '\t' << MPtrs.size1() << '\t' << sizeof(int) << '\t' << std::endl;

	#if PROFILE
	PROF_BEGIN(PROF_INIT_MARKING);
	#endif

//	Mh = (int*)malloc(sizemd);
	for (int i = 0; i < MPtrs.size1(); ++i) {
		//std::cout << "reaction " << i << " mval/pointers: ";
		for (int j = 0; j < 2; ++j) {
			if (MPtrs(i,j) != -1) {
				//std::cout << MPtrs(i,j) << ' ' << *MPtrs(i,j) << ' ';
				Mh[i*2 + j] = M(MPtrs(i,j), 0);
			}
			else {
				Mh[i*2 + j] = 0;
				//std::cout << MPtrs(i,j) << " NULL";
			}
		}
		//std::cout << std::endl;
	}

	#if PROFILE
	PROF_END(PROF_INIT_MARKING);
	#endif
	//for (int i = 0; i < MPtrs.size1(); ++i) {
		//std::cout << Mh[i*2 + 0] << ' ' << Mh[i*2 + 1] << std::endl;
	//}

	#if PROFILE
	PROF_BEGIN(PROF_CUDAMALLOC);
	#endif

//	cudaMalloc((void**) &Md, sizemd);
//	cudaMalloc((void**) &cd, sizec);
//	cudaMalloc((void**) &HFuncd, sizehfunc);
//
//	cudaMalloc((void**) &Hd, sizeh);

	#if PROFILE
	PROF_END(PROF_CUDAMALLOC);
	#endif

	#if PROFILE
	PROF_BEGIN(PROF_CUDAMEMCOPY_TO);
	#endif

	cudaMemcpy(Md, Mh, sizemd, cudaMemcpyHostToDevice);
	cudaMemcpy(cd, &(c.data()[0]), sizec, cudaMemcpyHostToDevice);
	cudaMemcpy(HFuncd, &(HFunc.data()[0]), sizehfunc, cudaMemcpyHostToDevice);

	#if PROFILE
	PROF_END(PROF_CUDAMEMCOPY_TO);
	#endif

	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(H.size1()/1024 + 1, 1, 1);

	#if PROFILE
	PROF_BEGIN(PROF_UPDATE_KERNEL);
	#endif

	UpdateKernel<<<dimGrid, dimBlock>>>(Md, cd, HFuncd, Hd, H.size1());
	cudaDeviceSynchronize();

	#if PROFILE
	PROF_END(PROF_UPDATE_KERNEL);
	#endif

	#if PROFILE
	PROF_BEGIN(PROF_CUDAMEMCOPY_FROM);
	#endif

	cudaMemcpy(&(H.data()[0]), Hd, sizeh, cudaMemcpyDeviceToHost);

	#if PROFILE
	PROF_END(PROF_CUDAMEMCOPY_FROM);
	#endif

	//std::cout << std::endl << "hazard: " << H << std::endl;
//	cudaFree(Md); cudaFree(cd); cudaFree(HFuncd); cudaFree(Hd);
//	free(Mh);

	H0 = prod(summer, H)(0,0);
	//std::cout << "H0 is: " << H0 << std::endl;
}

__global__ void UpdateKernel(int * Md, const FLOATT * cd, const rl_pointer * HFuncd, FLOATT * Hd, const int width) {
	int tx = threadIdx.x + (blockDim.x*blockIdx.x);


	if (tx < width) {
		//printf("Thread %d mvalues: %d %d\n", tx, Md[tx*2], Md[tx*2+1]);
		//printf("%d\t%.3f\t%ld\t%.3f\t%ld\t%ld\t%ld\t%ld\n", tx, cd[tx], (long long)HFuncd[tx], HFuncd[tx](5, 5, 5), (long long)prl_00, (long long)prl_10, (long long)prl_20, (long long)prl_11);
		//printf("%.8f %d %d\t", cd[tx], Md[tx*2], Md[tx*2+1]);
		//printf("%d %d\t", Md[tx*2], Md[tx*2+1]);
		//printf("%.3f ", HFuncd[tx](cd[tx], Md[tx*2], Md[tx*2+1]));
		//printf("%.3f ", Hd[tx]);
		Hd[tx] = HFuncd[tx](cd[tx], Md[tx*2], Md[tx*2+1]);
		//printf("%.3f %.3f\t", Hd[tx], HFuncd[tx](cd[tx], Md[tx*2], Md[tx*2+1]));
	}
}
