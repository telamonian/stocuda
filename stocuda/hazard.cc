/*
 * hazard.cc
 *
 *  Created on: Jan 9, 2013
 *      Author: tel
 */

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/irange.hpp>
#include <iostream>
#include <cmath>
#include <boost/function.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/algorithm/iteration/accumulate.hpp>
#include "hazard.hh"

#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

using namespace boost::numeric::ublas;

typedef matrix<double> state_type;
typedef double (*rl_pointer)(double, int *, int *);
typedef matrix<int>::iterator1 i1_t;
typedef matrix<int>::iterator2 i2_t;

double rl_00(double c, int * M1, int * M2);
double rl_10(double c, int * M1, int * M2);
double rl_11(double c, int * M1, int * M2);
double rl_20(double c, int * M1, int * M2);

void Hazard::Update(matrix<int> M) {
	static const matrix<double> summer (scalar_matrix<double> (1, H.size1(), 1));
	for (int i = 0; i<HFunc.size1(); ++i) {
		//std::cout << i << "\t" << H << "\t"  << Hfunc.size1() << "\t" << Hfunc.size2() << std::endl;
		H(i,0) = HFunc(i,0)(c(i,0), MPtrs(i,0), MPtrs(i,1));
	}
	H0 = std::abs(prod(summer, H)(0,0));
}

matrix<int> Hazard::InitOrder() {
	matrix<int> order(c.size1(), 4);
	int max1;
	int max2;
	int max1i;
	int max2i;
	for (matrix<int>::const_iterator1 it1 = Pre.begin1(); it1!=Pre.end1(); ++it1) {
		max1 = 0;
		max2 = 0;
		max1i = NULL;
		max2i = NULL;
		for (matrix<int>::const_iterator2 it2 = it1.begin(); it2!=it1.end(); ++it2) {
			if (*it2 > max1) {
				max2 = max1;
				max2i = max1i;
				max1 = *it2;
				max1i = it2.index2();
			}
			else if (*it2 > max2)
				max2 = *it2;
				max2i = it2.index2();
		}
		order(it1.index1(), 0) = max1;
		order(it1.index1(), 1) = max2;
		order(it1.index1(), 2) = max1i;
		order(it1.index1(), 3) = max2i;
	}
	return order;
}

matrix<int*> Hazard::InitMPtrs(matrix<int> &M) {
	matrix<int*> mptrs(c.size1(), 2);
	int max1;
	int max2;
	int max1i;
	int max2i;
	for (matrix<int>::const_iterator1 it1 = Pre.begin1(); it1!=Pre.end1(); ++it1) {
		max1 = 0;
		max2 = 0;
		max1i = NULL;
		max2i = NULL;
		for (matrix<int>::const_iterator2 it2 = it1.begin(); it2!=it1.end(); ++it2) {
			if (*it2 > max1) {
				max2 = max1;
				max2i = max1i;
				max1 = *it2;
				max1i = it2.index2();
			}
			else if (*it2 > max2)
				max2 = *it2;
				max2i = it2.index2();
		}
		if (max1 > 0)
			mptrs(it1.index1(), 0) = &M(max1i,0);
		else
			mptrs(it1.index1(), 0) = NULL;
		if (max2 > 0)
			mptrs(it1.index1(), 1) = &M(max2i,0);
		else
			mptrs(it1.index1(), 1) = NULL;
	}
	return mptrs;
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
		max1i = NULL;
		max2i = NULL;
		for (matrix<int>::const_iterator2 it2 = it1.begin(); it2!=it1.end(); ++it2) {
			if (*it2 > max1) {
				max2 = max1;
				max2i = max1i;
				max1 = *it2;
				max1i = it2.index2();
			}
			else if (*it2 > max2)
				max2 = *it2;
				max2i = it2.index2();
		}
		if (max1==0 && max2==0)
			hfunc(it1.index1(), 0) = rl_00;
		else if (max1==1 && max2==0)
			hfunc(it1.index1(), 0) = rl_10;
		else if (max1==1 && max2==1)
					hfunc(it1.index1(), 0) = rl_11;
		else if (max1==2 && max2==0)
					hfunc(it1.index1(), 0) = rl_20;
		else
			throw;
	}
	return hfunc;
}

double rl_00(double c, int * M1, int * M2) {
	return c;
}

double rl_10(double c, int * M1, int * M2) {
	return c*(*M1);
}

double rl_11(double c, int * M1, int * M2) {
	return c*(*M1)*(*M2);
}

double rl_20(double c, int * M1, int * M2) {
	return (c/2)*(*M1)*(*M1-1);
}
