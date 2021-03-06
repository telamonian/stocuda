/*
 * pnet.hh
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */

#ifndef HAZARD_HH_
#define HAZARD_HH_
#define FLOATT float

#include <cuda.h>
#include <pyublas/numpy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace pyublas;
using namespace boost::numeric::ublas;

typedef FLOATT (*rl_pointer)(FLOATT, int *, int *);

class Hazard {

public:

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
	Hazard(matrix<int>Prei, matrix<int> Posti, matrix<int> &Mi, matrix<FLOATT> ci):
		Pre(Prei),
		Post(Posti),
		c(ci),
		H(ci.size1(), 1),
		H0(0),
		S(InitS()),
		MPtrs(InitMPtrs(Mi)),
		HFunc(InitHFunc()) {
		InitGlobal(Mi);
		Update(Mi);
	}

	const matrix<int> Pre;
	const matrix<int> Post;
	const matrix<FLOATT> c;
	const matrix<FLOATT> S;
	matrix<FLOATT> H;
	FLOATT H0;
	matrix<int> Order;
	matrix<int> InitOrder();
	matrix<int> MPtrs;
	matrix<int> InitMPtrs(matrix<int> &M);
	matrix<rl_pointer> HFunc;
	matrix<rl_pointer> InitHFunc();
	int * Md;
	int ** MdPtrs;
	FLOATT * cd;
	rl_pointer * HFuncd;
	FLOATT * Hd;

	//void Update();
	void Update(matrix<int> &M);

	void InitGlobal(matrix<int> &M);

	matrix<FLOATT> InitS() {
		return trans(Post-Pre);
	}
};

#endif /* HAZARD_HH_ */
