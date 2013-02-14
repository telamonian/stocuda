# src directory

#include directories
INCLUDE_DIRECTORIES(.)

set(cudasource
	hazard.cu
	)

set(source 
	hazard.cc
	pnet.cc
	stocuda.cc
	)

#build and link
CUDA_ADD_LIBRARY(hazard ${cudasource}
				 OPTIONS -arch sm_30 -lineinfo)
PYTHON_ADD_MODULE(stocuda ${source})
    
target_link_libraries(stocuda
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        hazard
    )