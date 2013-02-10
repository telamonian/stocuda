# src directory

#include directories
INCLUDE_DIRECTORIES(.)

#glob (collect?) source files
FILE(GLOB source "*.cc")

#build and link
PYTHON_ADD_MODULE(stocuda ${source})
    
target_link_libraries(stocuda
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
    )