/*
 * pnet.hh
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */

#ifndef PNET_HH_
#define PNET_HH_

#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <time.h>
#include <string>
#include "hazard.hh"

using namespace pyublas;
using namespace boost::numeric::ublas;

class Pnet {

public:
	/// constructors
	/// default
	Pnet():
		//P(),
		//T(),
		Pre(),
		Post(),
		M(),
		c(),
		H(),
		t(0),
		A(),
		S(),
		mtgen(time(NULL)),
		dt(H.H0),
		uni(0, 1) {}

	/// empty, correctly dimension instance
	Pnet(int u, int v):
		//P(u, 1),
		//T(v, 1),
		Pre(v, u),
		Post(v, u),
		M(u,1),
		c(v,1),
		H(u, v),
		t(0),
		A(v, u),
		S(u, v),
		mtgen(time(NULL)),
		dt(H.H0),
		uni(0, 1) {}

	/// initialized from premade matrices
// 	Pnet(numpy_matrix<std::string> Pi, numpy_matrix<std::string> Ti, numpy_matrix<int> Prei, numpy_matrix<int> Posti, numpy_matrix<int> Mi, matrix<double> ci):
// 		P(Pi),
// 		T(Ti),
// 		Pre(Prei),
// 		Post(Posti),
// 		M(Mi),
// 		c(ci),
// 		H(Hazard(Pre, Post, M, c)),
// 		t(0),
// 		A(InitA()),
// 		S(InitS()),
// 		mtgen(time(NULL)),
// 		dt(H.H0),
// 		uni(0, 1) {}
		
    /// 'stubby' initializer to test pyublas
    Pnet(numpy_matrix<int> Prei, numpy_matrix<int> Posti, numpy_matrix<int> Mi, numpy_matrix<double> ci):
		//P(),
		//T(),
		Pre(Prei),
		Post(Posti),
		M(Mi),
		c(ci),
		t(0),
		A(InitA()),
		H(Hazard(Pre, Post, M, c)),
		S(InitS()),
		mtgen(time(NULL)),
		dt(H.H0),
		uni(0, 1) {}

	/// variables
	/// u x 1 column vector of species names
	//numpy_matrix<std::string> P;

	/// v x 1 column vector of reaction names
	//numpy_matrix<std::string> T;
	
	/// v x u pre matrix
	matrix<int> Pre;

	/// v x u post matrix
	matrix<int> Post;

	/// u x 1 column vector of current system marking
	matrix<int> M;

	/// v x 1 column vector of stochastic rate constants
	matrix<double> c;

	/// simulation time
	double t;

	/// v x u reaction matrix
	matrix<int> A;

	/// u x v stoichiometric matrix
	matrix<int> S;

	/// pointer to hazard matrix object
	Hazard H;

	/// mersenne twister. used to seed other random distributions
	boost::random::mt19937 mtgen;

	/// exponential distribution. used to calculate time inbetween Gillespie simulation steps
	boost::random::exponential_distribution<> dt;

	/// uniform distribution. used to calculate which transition occurs at each Gillespie step. Initialize with (0,1) to get standard uniform distribution
	boost::random::uniform_real_distribution<> uni;

	/// methods
	/// utility methods
//	double EWSum(matrix<double> m);

	/// initalization methods
	/// initialize reaction matrix
	matrix<int> InitA() {
		return (Post - Pre);
	}

	/// initialize stoichiometric matrix
	matrix<int> InitS() {
		return trans(A);
	}

	/// Hazard matrix methods

	/// Marking vector methods
	/// update marking via reaction represented in ith column of S, then update hazard matrix and sum
	void UpdateM(int i);

	/// Gillespie specific methods
	/// move the network n steps forward according to the Gillespie algorithm, record results by overwriting M
	void Gillespie(int n=1);
};

#endif /* PNET_HH_ */
