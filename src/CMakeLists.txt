cmake_minimum_required(VERSION 3.13)
find_package(hipSYCL)
find_package(MPI)

add_library(mylib STATIC bfs_custom.cpp)
target_include_directories(mylib PUBLIC /home/sbalint/hipSYCL-main/thesis/graph500/src)
target_include_directories(mylib PUBLIC /home/sbalint/hipSYCL-main/thesis/graph500/aml)
target_link_libraries(mylib PUBLIC MPI::MPI_C)
add_sycl_to_target(TARGET mylib)

add_executable(bfs_graph500 main.c utils.c validate.c ../aml/aml.c ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c csr_reference.c )
target_link_libraries(bfs_graph500 PUBLIC mylib)
target_link_libraries(bfs_graph500 PUBLIC MPI::MPI_C)
target_include_directories(bfs_graph500 PUBLIC /home/sbalint/hipSYCL-main/thesis/graph500/src)
target_include_directories(bfs_graph500 PUBLIC /home/sbalint/hipSYCL-main/thesis/graph500/aml)
target_link_directories(bfs_graph500 PUBLIC /opt/hipSYCL/boost/boost/lib)
target_link_directories(bfs_graph500 PUBLIC /home/sbalint/hipSYCL-main/dpc++/tbb/2021.2.0/lib/intel64/gcc4.8/)

target_link_options(bfs_graph500 PUBLIC -Wl,-rpath,/home/sbalint/hipSYCL-main/dpc++/tbb/2021.2.0/lib/intel64/gcc4.8/)
target_link_options(bfs_graph500 PUBLIC -Wl,-rpath=/opt/hipSYCL/boost/boost//lib)
target_link_options(bfs_graph500 PUBLIC -fopenmp)


#target_link_options(bfs_graph500 PUBLIC -ltbb)
target_link_libraries(bfs_graph500 PUBLIC tbb boost_context boost_fiber)


