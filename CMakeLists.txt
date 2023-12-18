cmake_minimum_required(VERSION 3.16)
project(mlp C)

set(CMAKE_C_STANDARD 11)

find_package(BLAS REQUIRED)
message(STATUS BLAS found: ${BLAS_LIBRARIES} ${BLAS_INCLUDE_DIRS})

add_executable(mlp main.c)
target_include_directories(mlp PRIVATE ${BLAS_INCLUDE_DIRS})
target_link_libraries(mlp PRIVATE ${BLAS_LIBRARIES} m)
