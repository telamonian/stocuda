#define FLOATT float
#define PROFILE 0

#include <pyublas/numpy.hpp>
#include "pnet.hh"

#if PROFILE
#include "Profile.h"
#include "ProfileCodes.h"
#endif

using namespace boost::python;
using namespace pyublas;

BOOST_PYTHON_MODULE(stocuda)
{
    class_<Pnet>("Pnet", init<numpy_matrix<int>,
            numpy_matrix<int>,
            numpy_matrix<int>,
            numpy_matrix<FLOATT> >())
        .def("Gillespie", &Pnet::Gillespie);
//    	.def("AddVecTest", &Pnet::AddVecTest);
}
