/*
 * pnet.hh
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */

#ifndef HAZARD_HH_
#define HAZARD_HH_

#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/function.hpp>

using namespace pyublas;
using namespace boost::numeric::ublas;

class Hazard {

public:
	typedef double (*rl_pointer)(double, int , int );

	/// constructors
	/// default
	Hazard():
		Pre(),
		Post(),
		c(),
		H(),
		H0(),
		S(),
		MPtrs(),
		HFunc(){}

	/// empty, correctly dimensioned instance
	Hazard(int u, int v):
		Pre(u,v),
		Post(u,v),
		c(v,1),
		H(v,1),
		H0(0),
		S(v,u),
		MPtrs(v, 1),
		HFunc(v, 1) {}

	/// initialized from premade matrices
	Hazard(matrix<int>Prei, matrix<int> Posti, matrix<int> &Mi, matrix<double> ci):
		Pre(Prei),
		Post(Posti),
		c(ci),
		H(ci.size1(), 1),
		H0(0),
		S(InitS()),
		MPtrs(InitMPtrs(Mi)),
		HFunc(InitHFunc()) {
		Update(Mi);}

	const matrix<int> Pre;
	const matrix<int> Post;
	const matrix<double> c;
	const matrix<double> S;
	matrix<double> H;
	double H0;
	matrix<int> Order;
	matrix<int> InitOrder();
	matrix<int> MPtrs;
	matrix<int> InitMPtrs(matrix<int> &M);
	matrix<rl_pointer> HFunc;
	matrix<rl_pointer> InitHFunc();

	//void Update();
	void Update(matrix<int> M);

	matrix<double> InitS() {
		return trans(Post-Pre);
	}
};

#endif /* HAZARD_HH_ */
