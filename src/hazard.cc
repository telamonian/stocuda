/*
 * hazard.cc
 *
 *  Created on: Jan 9, 2013
 *      Author: tel
 */
#define FLOATT float

#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <cmath>
#include "hazard.hh"

using namespace pyublas;
using namespace boost::numeric::ublas;

typedef FLOATT (*rl_pointer)(FLOATT, int , int );
typedef matrix<int>::iterator1 i1_t;
typedef matrix<int>::iterator2 i2_t;

//FLOATT rl_00(FLOATT c, int * M1, int * M2);
//FLOATT rl_10(FLOATT c, int * M1, int * M2);
//FLOATT rl_11(FLOATT c, int * M1, int * M2);
//FLOATT rl_20(FLOATT c, int * M1, int * M2);

//void Hazard::Update(matrix<int> M) {
//	static const scalar_matrix<FLOATT> summer (scalar_matrix<FLOATT> (1, H.size1(), 1));
//
//	for (int i = 0; i<HFunc.size1(); ++i) {
//	    //std::cout << MPtrs(i, 0) << '\t';
//		//std::cout << i << "\t" << H << std::endl;
//		H(i,0) = HFunc(i,0)(c(i,0), MPtrs(i,0), MPtrs(i,1));
//	}
//	//std::cout << std::endl;
//	H0 = prod(summer, H)(0,0);
//
//}

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
			else if (*it2 > max2) {
				max2 = *it2;
				max2i = it2.index2();
			}
		}
		order(it1.index1(), 0) = max1;
		order(it1.index1(), 1) = max2;
		order(it1.index1(), 2) = max1i;
		order(it1.index1(), 3) = max2i;
	}
	return order;
}


