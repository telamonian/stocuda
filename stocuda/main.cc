/*
 * main.cc
 *
 *  Created on: Jan 3, 2013
 *      Author: tel
 */

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <string>
#include "pnet.hh"
#include "hazard.hh"

int main () {
	matrix<std::string> P(2,1);
	matrix<std::string> T(2,1);
	matrix<int> Pre(2,2);
	matrix<int> Post(2,2);
	matrix<int> M(2,1);
	state_type c(2,1);

	P(0,0) = "U"; P(1,0) = "U2";
	T(0,0) = "Dimerisation"; T(1,0) = "Dissociation";
	Pre(0,0) = 2; Pre(0,1) = 0; Pre(1,0) = 0; Pre(1,1) = 1;
	Post(0,0) = 0; Post(0,1) = 2; Post(1,0) = 1; Post(1,1) = 0;
	M(0,0) = 1000; M(1,0) = 0;
	c(0,0) = 1.0; c(1,0) = .5;
	Pnet N(P, T, Pre, Post, M, c);
	std::cout << "time" << '\t' << "marking" << std::endl;
	std::cout << N.t << '\t' << N.M << std::endl;
	N.Gillespie(1000);
	return 0;
}
