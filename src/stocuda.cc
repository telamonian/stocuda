#include <pyublas/numpy.hpp>
#include "pnet.hh"

using namespace boost::python;
using namespace pyublas;

BOOST_PYTHON_MODULE(stocuda)
{
    class_<Pnet>("Pnet", init<numpy_matrix<int>,
            numpy_matrix<int>,
            numpy_matrix<int>,
            numpy_matrix<double> >())
        .def("Gillespie", &Pnet::Gillespie);
//    	.def("AddVecTest", &Pnet::AddVecTest);
}
