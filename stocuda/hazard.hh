/*
 * pnet.hh
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */

#ifndef HAZARD_HH_
#define HAZARD_HH_

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/function.hpp>

using namespace boost::numeric::ublas;

class Hazard {

public:
	typedef matrix<double> state_type;
	typedef double (*rl_pointer)(double, int *, int *);

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
	Hazard(matrix<int>Prei, matrix<int> Posti, matrix<int> &Mi, state_type ci):
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
	const state_type c;
	const state_type S;
	state_type H;
	double H0;
	matrix<int> Order;
	matrix<int> InitOrder();
	matrix<int*> MPtrs;
	matrix<int*> InitMPtrs(matrix<int>&);
	matrix<rl_pointer> HFunc;
	matrix<rl_pointer> InitHFunc();
	void Update(matrix<int> M);

	state_type InitS() {
		return trans(Post-Pre);
	}
};

#endif /* HAZARD_HH_ */
