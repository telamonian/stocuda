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

struct push_back_state_and_time
{
    std::vector<state_type>& m_states;
    std::vector<double>& m_times;

    push_back_state_and_time(std::vector<state_type> &states , std::vector<double> &times )
    :m_states(states) , m_times(times) {}

    void operator()(const state_type &x , double t)
    {
    	//std::cout << t << '\t' << x(0,0) << '\t' << x(1,0) << '\n';
        m_states.push_back(x);
        m_times.push_back(t);
    }
};

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

//double Pnet::EWSum(matrix<double> m) {
//		double sum = 0;
//		for (matrix<double>::iterator1 i1 = m.begin1(); i1!=m.end1(); ++i1)
//			for (matrix<double>::iterator2 i2 = i1.begin(); i2!=i1.end(); ++i2)
//				sum += *i2;
//		return sum;
//	}

void Pnet::UpdateM(int i) {
	M = M + project(S, range(0,M.size1()), range(i,i+1)); //column(S, i);
	H.Update(M);
}

void Pnet::Deterministic() {
//	state_type Mstd(M.size());
//	for (int i = 0; i<M.size(); ++i) {
//		Mstd[i] = M[i];
//	}
	///adaptive integration
	//typedef runge_kutta_cash_karp54< matrix<double> > error_stepper_type;
	//typedef controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
	//controlled_stepper_type controlled_stepper;
	//std::size_t steps = integrate_adaptive(controlled_stepper , *H, Mstd, 0.0, 1000.0, 0.00000001, push_back_state_and_time(x_vec, times));

	///convenience function integration
	std::size_t steps = integrate( H , M , 0.0 , .1 , .0000001, push_back_state_and_time(x_vec, times) );
	///std::cout << t << '\t' << x_vec << std::endl;

	///constant step integration
//	runge_kutta4< state_type > stepper;
//	std::size_t steps = integrate_const(stepper , *H, M, 0.0, 1, 0.00001, push_back_state_and_time(x_vec, times));

	for( std::size_t i=0; i<=steps; i++ )
	{
		std::cout << times[i] << '\t' << x_vec[i](0,0) << '\t' << x_vec[i](1,0) << '\n';
	}
}

