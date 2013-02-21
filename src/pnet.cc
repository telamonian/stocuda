/*
 * pnet.cc
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */
#define FLOATT float
#define PROFILE 1

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <time.h>
#include "pnet.hh"

#if PROFILE
#include "Profile.h"
#include "ProfileCodes.h"
#endif

using namespace boost::numeric::ublas;

void Pnet::Gillespie(int n) {
	FLOATT chance;
	FLOATT tsum;

	#if PROFILE
	PROF_INIT;
	#endif

	int c = 0;
	int j;
	for (int i = 0; i < n; ++i) {
		++c;
		t = t + dt(mtgen);
		chance = uni(mtgen)*H.H0;
		j = 0;
		tsum  = H.H(j,0);
		//std::cout << H->H << '\t' << H->H0 << std::endl;
		while (tsum < chance) {
			++j;
			tsum += H.H(j,0);
		}
		//std::cout << "reaction #" << j << " fired" << std::endl;
		PROF_BEGIN(PROF_UPDATEM);
		UpdateM(j);
		PROF_END(PROF_UPDATEM);
		//for (int k = 0; k < M.size1(); ++k)
		    //std::cout << &(M.data()[0]) + k << '\t' << *(&(M.data()[0]) + k) << '\t' ;
		//std::cout << std::endl;

		/// prep random number generators for next round
		dt.param(H.H0);
	}
	std::cout << c << '\t' << t << '\t' << M << std::endl;

	#if PROFILE
	PROF_WRITE;
	#endif
}

void Pnet::UpdateM(int i) {
	M = M + project(S, range(0,M.size1()), range(i,i+1)); //column(S, i);
	//std::cout << "pointers to pnet M:\n";
	//for (i=0;i<M.size1();++i){
	//	std::cout << &(M.data()[0]) + i << ' ' << M(i,0) << std::endl;
	//}
	H.Update(M);
}

