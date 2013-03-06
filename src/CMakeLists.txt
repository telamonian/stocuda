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
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    CUDA_ADD_LIBRARY(hazard SHARED ${cudasource}
				 OPTIONS -arch sm_30 -lineinfo)
	PYTHON_ADD_MODULE(stocuda SHARED ${source})

ELSE(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	CUDA_ADD_LIBRARY(hazard ${cudasource}
					 OPTIONS -arch sm_30 -lineinfo)
	PYTHON_ADD_MODULE(stocuda ${source})

ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    
target_link_libraries(stocuda
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        hazard
    )