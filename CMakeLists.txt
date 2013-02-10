cmake_minimum_required(VERSION 2.8.3)
project(hello)
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost)
FIND_PACKAGE(Boost COMPONENTS python)

# project options
OPTION(INCLUDE_CUDA "Set to OFF to no search for CUDA" ON)

#find cuda
IF(INCLUDE_CUDA)
    FIND_PACKAGE(CUDA)

    IF(CUDA_FOUND)
        MESSAGE("CUDA has been found")
    ELSE(CUDA_FOUND)
        MESSAGE(FATAL_ERROR "CUDA could not be found")
    ENDIF(CUDA_FOUND)
    
ENDIF(INCLUDE_CUDA)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREAD ON)

include_directories(/Library/Python/2.7/site-packages/numpy-1.8.0.dev_5c944b9_20120828-py2.7-macosx-10.8-x86_64.egg/numpy/core/include/)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})

#add subdirectories
ADD_SUBDIRECTORY(src)