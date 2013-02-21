#define PROFILE 0
#define BLOCKSIZE 1024
#define FLOATT float

#include <cuda.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "hazard.hh"

#if PROFILE
#include "Profile.h"
#include "ProfileCodes.h"
PROF_ALLOC;
#endif



using namespace boost::numeric::ublas;

//typedef FLOATT (*rl_pointer)(FLOATT, int *, int *);

__device__ int counter = 0;

__global__ void UpdateKernel(int * Md, int ** MdPtrs, FLOATT * cd, rl_pointer * HFuncd, FLOATT * Hd, int width);
__global__ void InitMdPtrs(int * Md, int * MdIndex, int ** MdPtrs, int width);
__device__ FLOATT rl_00(FLOATT c, int * M1, int * M2);
__device__ FLOATT rl_10(FLOATT c, int * M1, int * M2);
__device__ FLOATT rl_11(FLOATT c, int * M1, int * M2);
__device__ FLOATT rl_20(FLOATT c, int * M1, int * M2);

__device__ FLOATT rl_00(FLOATT c, int * M1, int * M2) {
	//printf("rl_00\n");
	return c;
}

__device__ FLOATT rl_10(FLOATT c, int * M1, int * M2) {
	//printf("rl_10 %d\n", *M1);
	return c*(*M1);
}

__device__ FLOATT rl_11(FLOATT c, int * M1, int * M2) {
    //printf("rl_11 %d\t%d\n", *M1, *M2);
	return c*(*M1)*(*M2);
}

__device__ FLOATT rl_20(FLOATT c, int * M1, int * M2) {
	//printf("rl_20 %d\n", *M1);
	return (c/2)*(*M1)*((*M1)-1);
}

const __device__ rl_pointer prl_00 = rl_00;
const __device__ rl_pointer prl_10 = rl_10;
const __device__ rl_pointer prl_20 = rl_20;
const __device__ rl_pointer prl_11 = rl_11;

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

void Hazard::InitGlobal(matrix<int> &M) {
	int sizemd;
	int sizec;
	int sizehfunc;
	int sizeh;

	int sizemptrs = c.size1() * 2 * sizeof(int);
	int sizemdptrs = c.size1() * 2 * sizeof(int*);
	sizemd = M.size1() * sizeof(int);
	sizec = c.size1() * sizeof(FLOATT);
	sizehfunc = HFunc.size1() * sizeof(rl_pointer);
	sizeh = H.size1() * sizeof(FLOATT);

	cudaMalloc((void**) &Md, sizemd);
	cudaMalloc((void**) &MdPtrs, sizemdptrs);
	cudaMalloc((void**) &cd, sizec);
	cudaMalloc((void**) &HFuncd, sizehfunc);
	cudaMalloc((void**) &Hd, sizeh);

	cudaMemcpy(Md, &(M.data()[0]), sizemd, cudaMemcpyHostToDevice);

//	for (int i = 0;i < 46;++i) {
//		printf("HFunc pointer for reaction %d: %0x\n", i, HFunc(i,0));
//	}


	int * MdIndex;
	cudaMalloc((void**) &MdIndex, sizemptrs);
	cudaMemcpy(MdIndex, &(MPtrs.data()[0]), sizemptrs, cudaMemcpyHostToDevice);
//	std::cout << "MPtrs: " << MPtrs << std::endl;

	dim3 dimBlock(BLOCKSIZE, 1, 1);
	dim3 dimGrid((c.size1()*2)/BLOCKSIZE + 1, 1, 1);
	InitMdPtrs<<<dimGrid, dimBlock>>>(Md, MdIndex, MdPtrs, c.size1()*2);
	//cudaDeviceSynchronize();

	cudaFree(MdIndex);
	cudaMemcpy(cd, &(c.data()[0]), sizec, cudaMemcpyHostToDevice);
	cudaMemcpy(HFuncd, &(HFunc.data()[0]), sizehfunc, cudaMemcpyHostToDevice);
}

__global__ void InitMdPtrs(int * Md, int * MdIndex, int ** MdPtrs, int width) {
	int tx = threadIdx.x + (blockDim.x*blockIdx.x);
	//for (int i=0;i<16;++i) {
		//printf("%d\n", Md[i]);
	//}
	//printf("%d %d\n", Md, MdPtrs);
	if (tx < width) {
		if (MdIndex[tx] >= 0) {
			MdPtrs[tx] = &(Md[MdIndex[tx]]);
//			printf("device species pointer %d: %ld\n", tx, (long long)(&(Md[MdIndex[tx]])));
		}
		else {
			MdPtrs[tx] = NULL;
		}
	}
}



void Hazard::Update(matrix<int> &M) {
	static const scalar_matrix<FLOATT> summer (scalar_matrix<FLOATT> (1, H.size1(), 1));

	int sizemd;
	int sizec;
	int sizehfunc;
	int sizeh;

	int sizemptrs = c.size1() * 2 * sizeof(int);
	int sizemdptrs = c.size1() * 2 * sizeof(int*);
	sizemd = M.size1() * sizeof(int);
	sizec = c.size1() * sizeof(FLOATT);
	sizehfunc = HFunc.size1() * sizeof(rl_pointer);
	sizeh = H.size1() * sizeof(FLOATT);

	#if PROFILE
	PROF_BEGIN(PROF_CUDAMEMCOPY_TO);
	#endif

	cudaMemcpy(Md, &(M.data()[0]), sizemd, cudaMemcpyHostToDevice);
//	cudaMemcpy(cd, &(c.data()[0]), sizec, cudaMemcpyHostToDevice);
//	cudaMemcpy(HFuncd, &(HFunc.data()[0]), sizehfunc, cudaMemcpyHostToDevice);

	#if PROFILE
	PROF_END(PROF_CUDAMEMCOPY_TO);
	#endif

	dim3 dimBlock(BLOCKSIZE, 1, 1);
	dim3 dimGrid(H.size1()/BLOCKSIZE + 1, 1, 1);

	#if PROFILE
	PROF_BEGIN(PROF_UPDATE_KERNEL);
	#endif

	UpdateKernel<<<dimGrid, dimBlock>>>(Md, MdPtrs, cd, HFuncd, Hd, H.size1());
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

//	std::cout << std::endl << "hazard: " << H << std::endl;
//	cudaFree(Md); cudaFree(cd); cudaFree(HFuncd); cudaFree(Hd);
//	free(Mh);

	H0 = prod(summer, H)(0,0);
//	std::cout << "H0 is: " << H0 << std::endl;
}

__global__ void UpdateKernel(int * Md, int ** MdPtrs, FLOATT * cd, rl_pointer * HFuncd, FLOATT * Hd, int width) {
	int tx = threadIdx.x + (blockDim.x*blockIdx.x);
//	int a=1, b=1;
//	int * pa = &a, * pb = &b;

	//printf("pointer to Md and MdPtrs cast as longs: %0x %0x\n", icMd, icMdPtrs);
	//printf("pointer to Md and MdPtrs cast as longs cast as pointers: %0x %0x\n", (int*)icMd, (int**)icMdPtrs);
//	int * Md;
//	int ** MdPtrs;
//	long recastMd;
//	long recastMdPtrs;
//	recastMd = (long)(int*)icMd;
//	recastMdPtrs = (long)(int**)icMdPtrs;
//	Md = (int*)icMd;
//	MdPtrs = (int**)icMdPtrs;

	//printf("pointer to Md and MdPtrs cast as longs cast as pointers cast as longs: %0x %0x\n", recastMd, recastMdPtrs);
	//printf("counter: %d", counter);
	//++counter;
	//printf("Thread %d width: %d\n", tx, width);
	//printf("Thread %d md mdptrs pointer: %0x %0x\n", tx, Md, MdPtrs);
//	if (tx < 1) {
//		printf("size of int: %d\n", sizeof(int));
//		for (int i = 0; i < 32; ++i) {
//			printf ("species %d marking: %d\n", i, Md[i]);
//		}
//		for (int i = 0; i < 46; ++i) {
//					printf ("reaction %d constant: %.3f\n", i, cd[i]);
//		}
//	}
	if (tx < width) {
//		if (HFuncd[tx]==prl_11) {
//			if (MdPtrs[tx*2] != NULL && MdPtrs[tx*2 + 1] != NULL) {
//				printf("Thread %d m values: %d %d\n", tx,*(MdPtrs[tx*2]), *(MdPtrs[tx*2 + 1]));
//				a = *(MdPtrs[tx*2]);
//				b = *(MdPtrs[tx*2+1]);
//				pa = &a;
//				pb = &b;
//			}
//			else if (MdPtrs[tx*2] != NULL) {
//				printf("Thread %d m values: %d NULL\n", tx,*(MdPtrs[tx*2]));
//				a = *(MdPtrs[tx*2]);
//				b = 10101010;
//				pa = &a;
//				pb = &b;
//			}
//			else {
//				printf("Thread %d m values: NULL NULL\n", tx);
//				a = 10101010;
//				b = 10101010;
//				pa = &a;
//				pb = &b;
//			}
//		}
		//printf("%.3f %.3f\t", Hd[tx], HFuncd[tx](cd[tx], MdPtrs[tx*2], MdPtrs[tx*2+1]));
		//printf("%d\t%.3f\t%0x\t%ld\t%ld\t%ld\t%ld\n", tx, cd[tx], (long long)HFuncd[tx], (long long)prl_00, (long long)prl_10, (long long)prl_20, (long long)prl_11);
		//printf("%.8f %d %d\t", cd[tx], Md[tx*2], Md[tx*2+1]);
		//printf("%d %d\t", Md[tx*2], Md[tx*2+1]);
		//printf("%.3f ", HFuncd[tx](cd[tx], Md[tx*2], Md[tx*2+1]));
		//printf("thread %d hfunc pointer: %0x\n ", tx, Hd[tx]);
		//Hd[tx] = 1;
		Hd[tx] = HFuncd[tx](cd[tx], MdPtrs[tx*2], MdPtrs[tx*2 + 1]);
		//Hd[tx] = HFuncd[tx](.1, pa, pb);
	}
}
