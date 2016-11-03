# Check cmake version.
cmake_minimum_required (VERSION 2.8)

# Project specification.
project (HMLP)
set (HMLP_VERSION_MAJOR 1)
set (HMLP_VERSION_MINOR 1)
set (HMLP_VERSION_PATCH 0)
set (HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})



# Compiler Options (HMLP can use Intel or GNU compilers.)
if ($ENV{HMLP_USE_INTEL} MATCHES "true")
  FIND_PROGRAM(ICC_PATH "icc")
  FIND_PROGRAM(ICPC_PATH "icpc")
  set (CMAKE_C_COMPILER   ${ICC_PATH} )
  set (CMAKE_CXX_COMPILER ${ICPC_PATH})
  set (HMLP_CFLAGS          "-std=c++11 -O3 -qopenmp -mavx -Werror -Wall -pedantic")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp")
else ($ENV{HMLP_USE_INTEL} MATCHES "true")
  set (CMAKE_C_COMPILER   gcc)
  set (CMAKE_CXX_COMPILER g++)
  set (HMLP_CFLAGS          "-std=c++11 -O3 -fopenmp -mavx")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")
endif ($ENV{HMLP_USE_INTEL} MATCHES "true")



# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")

# Turn on testing.
ENABLE_TESTING()

# Decide architecture
set (HMLP_ARCH_MAJOR $ENV{HMLP_ARCH_MAJOR})
set (HMLP_ARCH_MINOR $ENV{HMLP_ARCH_MINOR})
set (HMLP_ARCH ${HMLP_ARCH_MAJOR}/${HMLP_ARCH_MINOR})

set (HMLP_GPU_ARCH_MAJOR $ENV{HMLP_GPU_ARCH_MAJOR})
set (HMLP_GPU_ARCH_MINOR $ENV{HMLP_GPU_ARCH_MINOR})
set (HMLP_GPU_ARCH ${HMLP_GPU_ARCH_MAJOR}/${HMLP_GPU_ARCH_MINOR})

# GPU Compiler ptions
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  find_package (CUDA REQUIRED)
  include (FindCUDA)
  list (APPEND CUDA_NVCC_FLAGS "-O3")
  list (APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  list (APPEND CUDA_NVCC_FLAGS -DHMLP_USE_CUDA)
  set (CUDA_PROPAGATE_HOST_FLAGS "false")
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")



if ($ENV{HMLP_USE_VML} MATCHES "true")
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_VML")
  endif ($ENV{HMLP_USE_INTEL} MATCHES "true")
endif ($ENV{HMLP_USE_VML} MATCHES "true")

if ($ENV{HMLP_MIC_AVX512} MATCHES "true")
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (HMLP_CFLAGS          "${HMLP_CFLAGS} -xMIC-AVX512 -DHMLP_AVX512")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmemkind")
  endif ($ENV{HMLP_USE_INTEL} MATCHES "true")
endif ($ENV{HMLP_MIC_AVX512} MATCHES "true")

if ($ENV{HMLP_USE_BLAS} MATCHES "true")
  set(BLA_STATIC on)
  find_package (BLAS REQUIRED)
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_BLAS")
endif ($ENV{HMLP_USE_BLAS} MATCHES "true")
set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS} ${HMLP_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${HMLP_CFLAGS}")

# Linking infos.
if (($ENV{HMLP_USE_BLAS} MATCHES "true") OR ($ENV{HMLP_USE_VML} MATCHES "true"))
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
  else ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  endif ($ENV{HMLP_USE_INTEL} MATCHES "true")
endif (($ENV{HMLP_USE_BLAS} MATCHES "true") OR ($ENV{HMLP_USE_VML} MATCHES "true"))


# Software dependencies.
set (MKL_DIR $ENV{HMLP_MKL_DIR})


# Headers.
include_directories ("include" "frame" "${MKL_DIR}/include")
include_directories ( "kernel/reference" "kernel/${HMLP_ARCH}" )

if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  include_directories ("/usr/local/cuda/include" "${CMAKE_SOURCE_DIR}/kernel/${HMLP_GPU_ARCH}")
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")

file (GLOB HMLP_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${HMLP_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB FRAME_CU_SRC ${CMAKE_SOURCE_DIR}/frame/*.cu)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CXX_SRC ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CU_SRC ${CMAKE_SOURCE_DIR}/package/${HMLP_GPU_ARCH}/*.cu)


# Build the static library.
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  cuda_add_library (hmlp ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC}
    ${PACKAGE_CXX_SRC} ${FRAME_CU_SRC} ${PACKAGE_CU_SRC})
else ($ENV{HMLP_USE_CUDA} MATCHES "true")
  add_library (hmlp ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC} ${PACKAGE_CXX_SRC})
  if (($ENV{HMLP_USE_BLAS} MATCHES "true") AND NOT ($ENV{HMLP_USE_INTEL} MATCHES "true"))
    target_link_libraries (hmlp ${BLAS_LIBRARIES})
  endif ()
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")


# Build the executable files.
add_executable (test_hmlp.x ${CMAKE_SOURCE_DIR}/test/test_hmlp.cpp)
target_link_libraries(test_hmlp.x hmlp)
add_executable (test_gkmx.x ${CMAKE_SOURCE_DIR}/test/test_gkmx.cpp)
target_link_libraries(test_gkmx.x hmlp)
add_executable (test_strassen.x ${CMAKE_SOURCE_DIR}/test/test_strassen.cpp)
target_link_libraries(test_strassen.x hmlp)
add_executable (test_conv_relu_pool.x ${CMAKE_SOURCE_DIR}/test/test_conv_relu_pool.cpp)
target_link_libraries(test_conv_relu_pool.x hmlp)


if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  cuda_add_executable (test_gkmm_gpu.x ${CMAKE_SOURCE_DIR}/test/test_gkmm_gpu.cu)
  target_link_libraries(test_gkmm_gpu.x hmlp)
  cuda_add_cublas_to_target(test_gkmm_gpu.x)
  cuda_add_executable (test_strassen_gpu.x ${CMAKE_SOURCE_DIR}/test/test_strassen_gpu.cu)
  target_link_libraries(test_strassen_gpu.x hmlp)
  cuda_add_cublas_to_target(test_strassen_gpu.x)
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")





# add_executable (test_dgsks_list.x ${CMAKE_SOURCE_DIR}/test/test_dgsks_list.cpp)
# target_link_libraries(test_dgsks_list.x gsks)


# Install shell script
file (GLOB HMLP_SCRIPTS ${CMAKE_SOURCE_DIR}/test/*.sh)
install (FILES ${HMLP_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)


# Report
message ("===================================================")
message ("Version      =${HMLP_VERSION}")
message ("Source       =${CMAKE_SOURCE_DIR}")
message ("Target       =${CMAKE_BINARY_DIR}")
message ("HMLP_ARCH    =${HMLP_ARCH}")
message ("CC           =${CMAKE_C_COMPILER}")
message ("CFLAGS       =${CMAKE_C_FLAGS}")
message ("CXX          =${CMAKE_CXX_COMPILER}")
message ("CXXFLAGS     =${CMAKE_CXX_FLAGS}")
message ("Linker       =${CMAKE_EXE_LINKER_FLAGS}")
message ("HMLP_ARCH    =${HMLP_GPU_ARCH}")
message ("NVCCFLAGS    =${CUDA_NVCC_FLAGS}")
message ("MKLDIR       =${MKL_DIR}")
message ("===================================================")
