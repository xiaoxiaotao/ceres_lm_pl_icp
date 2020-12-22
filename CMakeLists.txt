cmake_minimum_required(VERSION 3.10)
project(pl_icp)
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -fopenmp -lopenblas -llapack")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g  -g -ggdb")




find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV 3 REQUIRED)
include_directories(${CERES_INCLUDE_DIRS}
        sophus)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(/home/tao/learning/ceres_icp_test/include/)


add_executable(pl_icp src/main.cpp src/pl_icp.cpp )
target_link_libraries (pl_icp ${CERES_LIBRARIES}  ${OpenCV_LIBRARIES}   ${PCL_LIBRARIES})