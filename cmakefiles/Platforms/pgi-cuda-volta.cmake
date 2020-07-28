### PGI Compiler for OpenMPI and OpenACC
set(TARGET_SUFFIX               ".pgi_acc")

set(ARCH                        "")
set(SIMD_SET                    "")
set(OPENMP_FLAGS                "-mp")
set(LAPACK_FLAGS                "-llapack -lblas")
set(ScaLAPACK_FLAGS             "-lscalapack -llapack -lblas")
set(ADDITIONAL_MACRO            "")
set(ADDITIONAL_OPTIMIZE_FLAGS   "-Mnoipa")

# search CUDA package
find_path(CUDA_TOOLKIT_ROOT_DIR "nvcc")
get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}/.." ABSOLUTE)

set(Fortran_FLAGS_General       "-Mpreprocess -acc -ta=tesla,cc70,ptxinfo,maxregcount:128 -Mcuda -Minfo=acc CUDAROOT=${CUDA_TOOLKIT_ROOT_DIR}")
set(C_FLAGS_General             "")

set(CMAKE_Fortran_COMPILER      "mpif90")
set(CMAKE_Fortran_FLAGS_DEBUG   "-pg")
set(CMAKE_Fortran_FLAGS_RELEASE "-fastsse")
set(CMAKE_C_COMPILER            "mpicc")
set(CMAKE_C_FLAGS_DEBUG         "-pg")
set(CMAKE_C_FLAGS_RELEASE       "-fastsse")


set(CUDA_TOOLKIT_INCLUDE "${CUDA_TOOLKIT_ROOT_DIR}/include")
set(CUDA_CUDART_LIBRARY  "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")


set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER        "g++")
set(CUDA_NVCC_FLAGS           "--generate-code arch=compute_70,code=sm_70 -Xptxas=-v")
set(CUDA_NVCC_FLAGS_DEBUG     "-lineinfo")
set(CUDA_NVCC_FLAGS_RELEASE   "-O3")


set(ENABLE_OPENACC      ON)
set(ENABLE_CUDA         ON)
set(USE_MPI             ON)
# set(EXPLICIT_VEC        ON)
# set(REDUCE_FOR_MANYCORE ON)

set(LARGE_BLOCKING ON)

########
# CMake Platform-specific variables
########
set(CMAKE_SYSTEM_NAME "Linux" CACHE STRING "Cross-compiling for NVIDIA with OpenACC")
set(CMAKE_SYSTEM_PROCESSOR "NVIDIA")
