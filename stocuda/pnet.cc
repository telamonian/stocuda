/*
 * pnet.cc
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <time.h>
#include "pnet.hh"

using namespace boost::numeric::ublas;
using namespace boost::numeric::odeint;

void Pnet::Gillespie(int n) {
	double chance;
	double tsum;
	int j;
	for (int i = 0; i < n; ++i) {
		t = t + dt(mtgen);
		chance = uni(mtgen)*H.H0;
		j = 0;
		tsum  = H.H(j,0);
		//std::cout << H->H << '\t' << H->H0 << std::endl;
		while (tsum < chance) {
			++j;
			tsum += H.H(j,0);
		}
		UpdateM(j);
		std::cout << t << '\t' << M << std::endl;

		/// prep random number generators for next round
		dt.param(H.H0);
	}
}

void Pnet::UpdateM(int i) {
	M = M + project(S, range(0,M.size1()), range(i,i+1)); //column(S, i);
	H.Update(M);
}

