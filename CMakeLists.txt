cmake_minimum_required(VERSION 2.9)
project(ndt_map_py)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

find_package(pybind11)
find_package(ndt_map)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.11.1 REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  ${ndt_map_INCLUDE_DIR}
  ${PCL_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# pybind modules
pybind11_add_module(ndt_map src/ndt_map_py/ndt_map.cpp)
target_link_libraries(ndt_map PRIVATE ${PYTHON_LIBRARIES} ${OpenCV_LIBRARIES} ${ndt_map_LIBRARIES} ${PCL_LIBRARIES})
set_target_properties(ndt_map PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX})

install(TARGETS DESTINATION ${CMAKE_INSTALL_PREFIX})
