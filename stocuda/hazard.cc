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
typedef matrix<int>::iterator1 i1_t;
typedef matrix<int>::iterator2 i2_t;

double rl_00(double c, int * M1, int * M2);
double rl_10(double c, int * M1, int * M2);
double rl_11(double c, int * M1, int * M2);
double rl_20(double c, int * M1, int * M2);

void Hazard::Update(state_type M) {
	static const matrix<double> summer (scalar_matrix<double> (1, H.size1(), 1));
	for (int i = 0; i<Hfunc.size1(); ++i) {
		//std::cout << i << "\t" << H << "\t"  << Hfunc.size1() << "\t" << Hfunc.size2() << std::endl;
		H(i,0) = Hfunc(i,0)(M);
	}
	H0 = std::abs(prod(summer, H)(0,0));
}
void Hazard::operator() ( const state_type &x , state_type &dxdt , const double /* t */ ) {
	double temp;
	for (int i = 0; i<x.size1(); ++i){
		temp = 0;
		for (int j = 0; j<S.size2(); ++j) {
			temp += S(i,j)*Hfunc(j,0)(x);
		}
		dxdt(i,0) = temp;
	}
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

matrix<int*> Hazard::InitMPtrs() {
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

matrix<rl_pointer> Hazard::InitHfunc() {
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

//matrix<boost::function<double(matrix<double>)> > Hazard::InitHfunc() {
//	matrix<boost::function<double(state_type)> > hfunc(c.size1(), 1);
//	double c_coeff;
//	double rl = 1;
//	for (matrix<int>::const_iterator1 it1 = Pre.begin1(); it1!=Pre.end1(); ++it1) {
//		c_coeff = boost::accumulate(it1.begin(), it1.end(), 1.0, [] (double c, int s) {
//			if (s > 1) {
//				for (int x: boost::irange(2, s+1)) {
//					c *= x;
//				}
//			}
//			return c;
//		});
//		hfunc(it1.index1(), 0) = [=, this, &rl] (state_type M) {
//			double c = (this->c)(it1.index1(), 0)*(1.0/c_coeff);
//			rl = 1;
//			for (matrix<int>::const_iterator2 it2 = it1.begin(); it2!=it1.end(); ++it2) {
//				if (*it2 > 0) {
//					boost::integer_range<int> ir = boost::irange(0, *it2);
//					rl *= boost::fusion::accumulate(ir.begin(), ir.end(), 1,[=, &M] (double rrl, int r) {
//						return rrl*(M(it2.index2(), 0) - r);
//					});
//				}
//			}
//			return c*rl;
//		};
//	}
//	return hfunc;
//}



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
