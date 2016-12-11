# Check cmake version.
cmake_minimum_required (VERSION 2.8)

# # Project specification.
# project (HMLP)
# set (HMLP_VERSION_MAJOR 1)
# set (HMLP_VERSION_MINOR 1)
# set (HMLP_VERSION_PATCH 0)
# set (HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})

if ( APPLE )
  set (CMAKE_MACOSX_RPATH TRUE)
endif ()


# Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")

# Decide architecture
set (HMLP_ARCH_MAJOR $ENV{HMLP_ARCH_MAJOR})
set (HMLP_ARCH_MINOR $ENV{HMLP_ARCH_MINOR})
set (HMLP_ARCH ${HMLP_ARCH_MAJOR}/${HMLP_ARCH_MINOR})

set (HMLP_GPU_ARCH_MAJOR $ENV{HMLP_GPU_ARCH_MAJOR})
set (HMLP_GPU_ARCH_MINOR $ENV{HMLP_GPU_ARCH_MINOR})
set (HMLP_GPU_ARCH ${HMLP_GPU_ARCH_MAJOR}/${HMLP_GPU_ARCH_MINOR})


# Project specification.
project (HMLP)
set (HMLP_VERSION_MAJOR 1)
set (HMLP_VERSION_MINOR 1)
set (HMLP_VERSION_PATCH 0)
set (HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})




# Compiler Options (HMLP can use Intel or GNU compilers.)
# ---------------------------
if ($ENV{HMLP_USE_INTEL} MATCHES "true")
  find_program (ICC_PATH   "icc")
  find_program (ICPC_PATH "icpc")
  set (CMAKE_C_COMPILER    ${ICC_PATH})
  set (CMAKE_CXX_COMPILER ${ICPC_PATH})
  set (HMLP_CFLAGS            "-std=c++11 -O3 -qopenmp -mavx -Werror -Wall -pedantic -fPIC")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp")
else ($ENV{HMLP_USE_INTEL} MATCHES "true")
  set (CMAKE_C_COMPILER   gcc)
  set (CMAKE_CXX_COMPILER g++)
  set (HMLP_CFLAGS            "-std=c++11 -O3 -fopenmp -mavx -fPIC")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")
endif ()


# Project specification.
# project (HMLP)
# set (HMLP_VERSION_MAJOR 1)
# set (HMLP_VERSION_MINOR 1)
# set (HMLP_VERSION_PATCH 0)
# set (HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})



# Remove -rdynamic
# ---------------------------
set (CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set (CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")


# Cross compilation for Android OS
# ---------------------------
if ($ENV{HMLP_ARCH_MAJOR} MATCHES "arm")
  set (CMAKE_C_COMPILER   aarch64-linux-android-gcc   )
  set (CMAKE_CXX_COMPILER aarch64-linux-android-g++   )
  set (CMAKE_AR           aarch64-linux-android-ar    )
  set (CMAKE_RANLIB       aarch64-linux-android-ranlib)
  #set (HMLP_CFLAGS "-std=c++11 -O3 -fopenmp -mfpu=neon -mfloat-abi=softfp -march=armv8-a -fPIC -fPIE")
  set (HMLP_CFLAGS "-std=c++11 -O3 -fopenmp -ftree-vectorize -mtune=cortex-a57.cortex-a53 -march=armv8-a+fp+simd -mcpu=cortex-a57.cortex-a53 -fPIC -fPIE")
  set (CMAKE_EXE_LINKER_FLAGS "-fopenmp -lm -pie -llog")
  set (CMAKE_OSX_SYSROOT    "")
  set (CMAKE_C_LINK_FLAGS   "")
  set (CMAKE_CXX_LINK_FLAGS "")
endif ()


# GPU Compiler sptions
# ---------------------------
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  find_package (CUDA REQUIRED)
  include (FindCUDA)
  list (APPEND CUDA_NVCC_FLAGS "-O3")
  list (APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  list (APPEND CUDA_NVCC_FLAGS -DHMLP_USE_CUDA)
  set (CUDA_PROPAGATE_HOST_FLAGS "false")
  set (CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif ()

# Turn on testing.
ENABLE_TESTING()





# Xeon Phi options
# ---------------------------
if ($ENV{HMLP_MIC_AVX512} MATCHES "true")
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (HMLP_CFLAGS          "${HMLP_CFLAGS} -xMIC-AVX512 -DHMLP_AVX512")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmemkind")
  endif ()
endif ()


# BLAS and VML options
# ---------------------------
if ($ENV{HMLP_USE_VML} MATCHES "true")
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_VML")
  endif ()
endif ()

if ($ENV{HMLP_USE_BLAS} MATCHES "true")
  set(BLA_STATIC on)
  # find_package (BLAS REQUIRED)
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_BLAS")
endif ()


# Pthread runtime
# ---------------------------
set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_PTHREAD_RUNTIME")


# Setup all flags
# ---------------------------
set (CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} ${HMLP_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${HMLP_CFLAGS}")


# Software dependencies.
# ---------------------------
set (MKL_DIR $ENV{HMLP_MKL_DIR})
set (QSML_DIR $ENV{HMLP_QSML_DIR})


# Linking infos.
# ---------------------------
if (($ENV{HMLP_USE_BLAS} MATCHES "true") OR ($ENV{HMLP_USE_VML} MATCHES "true"))
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
    include_directories ("${MKL_DIR}/include")
  else ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  endif ()
endif ()

if (($ENV{HMLP_USE_BLAS} MATCHES "true") AND ($ENV{HMLP_ARCH_MAJOR} MATCHES "arm"))
  if (DEFINED QSML_DIR)
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${QSML_DIR}/lib -lQSML-0.15.0 -lsymphony-cpu -lsymphonypower")
    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${QSML_DIR}/lib -lQSML-sequential-0.15.0 -lsymphony-cpu -lsymphonypower")
    include_directories ("${QSML_DIR}/include")
  endif()
endif()



# Headers.
# ---------------------------
include_directories ("include" "frame")
include_directories ( "kernel/reference" "kernel/${HMLP_ARCH}" )

if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  include_directories ("/usr/local/cuda/include" "${CMAKE_SOURCE_DIR}/kernel/${HMLP_GPU_ARCH}")
endif ()

file (GLOB HMLP_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${HMLP_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB FRAME_CU_SRC ${CMAKE_SOURCE_DIR}/frame/*.cu)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.cpp)
file (GLOB ASSEMBLY_SRC ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.s)
file (GLOB PACKAGE_CXX_SRC ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CU_SRC ${CMAKE_SOURCE_DIR}/package/${HMLP_GPU_ARCH}/*.cu)


#SET(CMAKE_ASM_COMPILE_OBJECT "<CMAKE_ASM_COMPILER> -o <OBJECT> <SOURCE>")


# Build the static library.
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  cuda_add_library (hmlp STATIC ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC}
    ${PACKAGE_CXX_SRC} ${FRAME_CU_SRC} ${PACKAGE_CU_SRC})
else ($ENV{HMLP_USE_CUDA} MATCHES "true")
  add_library (hmlp STATIC ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC}
    ${PACKAGE_CXX_SRC} ${ASSEMBLY_SRC})
  #if (($ENV{HMLP_USE_BLAS} MATCHES "true") AND NOT ($ENV{HMLP_USE_INTEL} MATCHES "true"))
  #  target_link_libraries (hmlp ${BLAS_LIBRARIES})
  #endif ()
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")


# add_library (dyhmlp SHARED ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC}
#     ${PACKAGE_CXX_SRC} ${ASSEMBLY_SRC})
# if (($ENV{HMLP_USE_BLAS} MATCHES "true") AND ($ENV{HMLP_USE_INTEL} MATCHES "true"))
#   target_link_libraries (dyhmlp -mkl=parallel)
# endif()



# Build the executable files.
# ---------------------------

#add_executable (hello_world.x ${CMAKE_SOURCE_DIR}/test/hello_world.cpp)
#target_link_libraries(hello_world.x hmlp)


# Build CNN test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/conv2d.cpp)
  add_executable (test_conv2d.x ${CMAKE_SOURCE_DIR}/test/test_conv2d.cpp)
  target_link_libraries(test_conv2d.x hmlp)
endif ()


# Build GSKS test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/gsks.cpp)
  add_executable (test_hmlp.x ${CMAKE_SOURCE_DIR}/test/test_hmlp.cpp)
  target_link_libraries(test_hmlp.x hmlp)
endif ()

# Build GKMX test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/gkmx.cpp)
  add_executable (test_gkmx.x ${CMAKE_SOURCE_DIR}/test/test_gkmx.cpp)
  target_link_libraries(test_gkmx.x hmlp)
endif ()

# Build STRASSEN test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/strassen.cpp)
  add_executable (test_strassen.x ${CMAKE_SOURCE_DIR}/test/test_strassen.cpp)
  target_link_libraries(test_strassen.x hmlp)
endif ()

# Build CONV_RELU_POOL test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/conv_relu_pool.cpp)
  add_executable (test_conv_relu_pool.x ${CMAKE_SOURCE_DIR}/test/test_conv_relu_pool.cpp)
  target_link_libraries(test_conv_relu_pool.x hmlp)
endif ()

# Build GSKNN test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/gsknn.cpp)
  add_executable (test_gsknn.x ${CMAKE_SOURCE_DIR}/test/test_gsknn.cpp)
  target_link_libraries(test_gsknn.x hmlp)
endif ()

# Build TREE test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/tree.cpp)
  add_executable (test_tree.x ${CMAKE_SOURCE_DIR}/test/test_tree.cpp)
  target_link_libraries(test_tree.x hmlp)
endif ()


# add_executable (test_dgsks_list.x ${CMAKE_SOURCE_DIR}/test/test_dgsks_list.cpp)
# target_link_libraries(test_dgsks_list.x gsks)



# Build GPU GKMX test suit
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  cuda_add_executable (test_gkmm_gpu.x ${CMAKE_SOURCE_DIR}/test/test_gkmm_gpu.cu)
  target_link_libraries(test_gkmm_gpu.x hmlp)
  cuda_add_cublas_to_target(test_gkmm_gpu.x)
  cuda_add_executable (test_strassen_gpu.x ${CMAKE_SOURCE_DIR}/test/test_strassen_gpu.cu)
  target_link_libraries(test_strassen_gpu.x hmlp)
  cuda_add_cublas_to_target(test_strassen_gpu.x)
endif ()






# Install shell script
# ---------------------------
file (GLOB HMLP_SCRIPTS ${CMAKE_SOURCE_DIR}/test/*.sh)
install (FILES ${HMLP_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)


# Report
# ---------------------------
message ("===================================================")
message ("Version      =${HMLP_VERSION}")
message ("Source       =${CMAKE_SOURCE_DIR}")
message ("Target       =${CMAKE_BINARY_DIR}")
message ("HMLP_ARCH    =${HMLP_ARCH}")
message ("CC           =${CMAKE_C_COMPILER}")
message ("CFLAGS       =${CMAKE_C_FLAGS}")
message ("C Linker     =${CMAKE_C_LINK_FLAGS}")
message ("CXX          =${CMAKE_CXX_COMPILER}")
message ("CXXFLAGS     =${CMAKE_CXX_FLAGS}")
message ("CXX Linker   =${CMAKE_CXX_LINK_FLAGS}")
message ("EXE Linker   =${CMAKE_EXE_LINKER_FLAGS}")
message ("AR           =${CMAKE_AR}")
message ("RANLIB       =${CMAKE_RANLIB}")
message ("HMLP_ARCH    =${HMLP_GPU_ARCH}")
message ("HOST_COMPILER=${CUDA_HOST_COMPILER}")
message ("NVCCFLAGS    =${CUDA_NVCC_FLAGS}")
message ("MKLDIR       =${MKL_DIR}")
message ("QSMLDIR      =${QSML_DIR}")
message ("BLAS_LIBRARY =${BLAS_LIBRARIES}")
message ("BLAS_LINKER  =${BLAS_LINKER_FLAGS}")
message ("===================================================")
