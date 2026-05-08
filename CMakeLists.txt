cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(linear_curvature_computation 
    src/main.cpp 
    src/halfedge2.cpp
)

target_include_directories(linear_curvature_computation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

