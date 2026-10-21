cmake_minimum_required(VERSION 3.20)
project(floodca LANGUAGES C CXX)

option(FCA_DOUBLE "Use double precision real" OFF)
option(FCA_WITH_OMP "Build OpenMP backend" ON)
option(FCA_WITH_MPI "Build MPI backends" ON)
option(FCA_WITH_CUDA "Build CUDA backends" ON)
option(FCA_BUILD_TESTS "Build C physics tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:C>:-Wall>)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
endif()

if(FCA_DOUBLE)
    add_compile_definitions(FCA_DOUBLE)
endif()

set(FCA_CORE_SRCS
    src/common/config.c
    src/common/io.c
    src/common/grid.c
    src/common/rain.c
    src/common/diag.c
    src/common/timer.c
    src/common/log.c
    src/common/model.c
    src/common/cli.c
)

add_library(floodca_core STATIC ${FCA_CORE_SRCS})
target_include_directories(floodca_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/kernels
)
target_link_libraries(floodca_core PUBLIC m)
target_compile_definitions(floodca_core PRIVATE _ISOC11_SOURCE)

add_executable(floodca_serial src/serial/main_serial.c)
target_link_libraries(floodca_serial PRIVATE floodca_core)
target_compile_definitions(floodca_serial PRIVATE _ISOC11_SOURCE)

if(FCA_WITH_OMP)
    find_package(OpenMP REQUIRED)
    add_library(floodca_core_omp STATIC ${FCA_CORE_SRCS})
    target_include_directories(floodca_core_omp PUBLIC
        ${CMAKE_SOURCE_DIR}/src/common
        ${CMAKE_SOURCE_DIR}/src/kernels
    )
    target_link_libraries(floodca_core_omp PUBLIC m OpenMP::OpenMP_C)
    target_compile_definitions(floodca_core_omp PRIVATE _ISOC11_SOURCE FCA_OMP)
    add_executable(floodca_omp src/omp/main_omp.c)
    target_link_libraries(floodca_omp PRIVATE floodca_core_omp OpenMP::OpenMP_C)
    target_compile_definitions(floodca_omp PRIVATE _ISOC11_SOURCE FCA_OMP)
endif()

if(FCA_WITH_MPI)
    find_package(MPI REQUIRED)
    add_executable(floodca_mpi src/mpi/main_mpi.c src/mpi/halo.c)
    target_include_directories(floodca_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src/mpi)
    target_link_libraries(floodca_mpi PRIVATE floodca_core MPI::MPI_C)
    target_compile_definitions(floodca_mpi PRIVATE FCA_WITH_MPI _ISOC11_SOURCE)
    if(FCA_WITH_OMP)
        add_executable(floodca_ens_mpi_omp src/hybrid/main_ens_mpi_omp.c src/hybrid/ens_common.c)
        target_link_libraries(floodca_ens_mpi_omp PRIVATE floodca_core_omp MPI::MPI_C OpenMP::OpenMP_C)
        target_compile_definitions(floodca_ens_mpi_omp PRIVATE FCA_WITH_MPI FCA_OMP _ISOC11_SOURCE)
    endif()
endif()

if(FCA_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    if(NOT CMAKE_CUDA_HOST_COMPILER AND CMAKE_C_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER} CACHE FILEPATH "" FORCE)
    endif()
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    # FMA contraction must be off on the device for the E1 equivalence build.
    option(FCA_CUDA_FMAD "Allow FMA contraction in device code" ON)
    foreach(tgt floodca_cuda floodca_cuda_timing)
        add_executable(${tgt} src/cuda/main_cuda.c src/cuda/kernels.cu)
        target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/src/cuda)
        target_link_libraries(${tgt} PRIVATE floodca_core)
        set_target_properties(${tgt} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
        if(NOT FCA_CUDA_FMAD)
            target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
        endif()
    endforeach()
    target_compile_definitions(floodca_cuda_timing PRIVATE FCA_CUDA_TIMING)
    if(FCA_WITH_MPI)
        add_executable(floodca_ens_mpi_cuda src/hybrid/main_ens_mpi_cuda.c src/hybrid/ens_common.c
                       src/cuda/kernels.cu)
        target_include_directories(floodca_ens_mpi_cuda PRIVATE ${CMAKE_SOURCE_DIR}/src/cuda)
        target_link_libraries(floodca_ens_mpi_cuda PRIVATE floodca_core MPI::MPI_C)
        target_compile_definitions(floodca_ens_mpi_cuda PRIVATE FCA_WITH_MPI _ISOC11_SOURCE)
        if(NOT FCA_CUDA_FMAD)
            target_compile_options(floodca_ens_mpi_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
        endif()
    endif()
endif()

if(FCA_BUILD_TESTS)
    add_executable(test_physics tests/c/test_physics.c)
    target_link_libraries(test_physics PRIVATE floodca_core)
    add_executable(test_mass tests/c/test_mass.c)
    target_link_libraries(test_mass PRIVATE floodca_core)
    add_executable(test_lake_at_rest tests/c/test_lake_at_rest.c)
    target_link_libraries(test_lake_at_rest PRIVATE floodca_core)
    add_executable(test_rain tests/c/test_rain.c)
    target_link_libraries(test_rain PRIVATE floodca_core)
endif()

install(TARGETS floodca_serial RUNTIME DESTINATION bin)
if(FCA_WITH_OMP)
    install(TARGETS floodca_omp RUNTIME DESTINATION bin)
endif()
