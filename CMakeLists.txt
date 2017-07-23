# Check cmake version.
# ---------------------------
cmake_minimum_required (VERSION 2.8)

# Project specification.
# ---------------------------
project (HMLP)
set (HMLP_VERSION_MAJOR 1)
set (HMLP_VERSION_MINOR 1)
set (HMLP_VERSION_PATCH 0)
set (HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})
ENABLE_TESTING()

# Cmake scripts and modules
# ---------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Decide architecture
# ---------------------------
set (HMLP_ARCH_MAJOR     $ENV{HMLP_ARCH_MAJOR})
set (HMLP_ARCH_MINOR     $ENV{HMLP_ARCH_MINOR})
set (HMLP_ARCH           ${HMLP_ARCH_MAJOR}/${HMLP_ARCH_MINOR})
set (HMLP_GPU_ARCH_MAJOR $ENV{HMLP_GPU_ARCH_MAJOR})
set (HMLP_GPU_ARCH_MINOR $ENV{HMLP_GPU_ARCH_MINOR})
set (HMLP_GPU_ARCH       ${HMLP_GPU_ARCH_MAJOR}/${HMLP_GPU_ARCH_MINOR})

# Detecting MacOS (mac cannot export $CC to change compilers)
# ---------------------------
if ( APPLE )
  message( STATUS "MacOS detected")
  set (CMAKE_MACOSX_RPATH TRUE)
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_C_COMPILER      icc)
    set (CMAKE_CXX_COMPILER    icpc)
    set (CMAKE_C_COMPILER_ID   Intel)
    set (CMAKE_CXX_COMPILER_ID Intel)
  else ()
    set (CMAKE_C_COMPILER      gcc)
    set (CMAKE_CXX_COMPILER    g++)
    set (CMAKE_C_COMPILER_ID   GNU)
    set (CMAKE_CXX_COMPILER_ID GNU)
  endif()
endif ()



# Software dependencies.
# ---------------------------
set (MKLROOT $ENV{MKLROOT})
set (OPENBLASROOT $ENV{OPENBLASROOT})
set (QSMLROOT $ENV{QSMLROOT})



# Compiler Options (HMLP can use Intel or GNU compilers.)
# ---------------------------
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

  # using Clang or AppleClang
  message( STATUS "Clang compiler." ) 
  message( WARNING "(Apple)Clang compiler is not supported." ) 
  message( FATAL_ERROR "You must have a generic GNU or Intel compiler." )

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

  # using GCC
  message( STATUS "GNU compiler." ) 
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -g -std=c++11 -O3 -fopenmp")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -fPIC")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")

  # try to detect and link BLAS/LAPACK
  if ($ENV{HMLP_USE_BLAS} MATCHES "true")

    find_package (BLAS REQUIRED)
    find_package (LAPACK REQUIRED)
    set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_BLAS")

    if ( OPENBLASROOT )
      message( STATUS "use OpenBLAS for BLAS/LAPACK" )
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L$ENV{OPENBLASROOT} -lopenblas")
    else ()
      if ( BLAS_LINKER_FLAGS )
        message( "${BLAS_LINKER_FLAGS}" )
        message( "${BLAS_LIBRARIES}" )
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
      else ()
        message( WARNING "cmake FindBLAS fails to generate a proper linker for BLAS" )
        message( FATAL_ERROR "fail to link to a BLAS library. Try to set HMLP_USE_BLAS=false." )  
      endif ()

      if ( LAPACK_LINKER_FLAGS )
        message("${LAPACK_LINKER_FLAGS}")
        message("${LAPACK_LIBRARIES}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
      else ()
        message( WARNING "cmake FindBLAS fails to generate a proper linker for LAPACK" )
        message( FATAL_ERROR "fail to link to a LAPACK library. Try to set HMLP_USE_BLAS=false." )  
      endif ()

    endif ()

  else ()
    message( WARNING "compile without BLAS/LAPACK. GOFMM will not be available.")
  endif ($ENV{HMLP_USE_BLAS} MATCHES "true")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")

  # using Intel C++
  message( STATUS "Intel compiler." )
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -g -std=c++11 -O3 -qopenmp")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -Werror -Wall -pedantic -fPIC")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -DUSE_INTEL")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp")

  # using mkl (still we use FindBLAS to detect Intel MKL) 
  if ($ENV{HMLP_USE_BLAS} MATCHES "true")
    set(ENV{BLA_VENDOR} "Intel10_64lp")
    find_package (BLAS REQUIRED)
    set(BLA_VENDOR "Intel")
    find_package (LAPACK REQUIRED)
    set (HMLP_CFLAGS            "${HMLP_CFLAGS} -DUSE_BLAS -DUSE_VML")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
    include_directories ("$ENV{MKLROOT}/include")
  else ()
    message( WARNING "compile without BLAS/LAPACK. GOFMM will not be available.")
  endif ($ENV{HMLP_USE_BLAS} MATCHES "true")

endif()



# Cross compilation for Android OS
# ---------------------------
if ($ENV{HMLP_ARCH_MAJOR} MATCHES "arm")
  set (CMAKE_C_COMPILER       aarch64-linux-android-gcc   )
  set (CMAKE_CXX_COMPILER     aarch64-linux-android-g++   )
  set (CMAKE_AR               aarch64-linux-android-ar    )
  set (CMAKE_RANLIB           aarch64-linux-android-ranlib)
  set (HMLP_CFLAGS            "-std=c++1y -O3 -fopenmp -fPIC -fPIE")
  set (CMAKE_EXE_LINKER_FLAGS "-fopenmp -lm -pie -llog")
  set (CMAKE_OSX_SYSROOT      "")
  set (CMAKE_C_LINK_FLAGS     "")
  set (CMAKE_CXX_LINK_FLAGS   "")

  ## using QSML
  if ($ENV{HMLP_USE_BLAS} MATCHES "true")
    if ( DEFINED QSMLROOT )
      set (QSML_LINKER_FLAGS "-L${QSMLROOT}/lib -lQSML-0.15.0 -lsymphony-cpu -lsymphonypower")
      #set (QSML_LINKER_FLAGS "-L${QSMLROOT}/lib -lQSML-sequential-0.15.0 -lsymphony-cpu -lsymphonypower")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${QSML_LINKER_FLAGS}")
      include_directories ("${QSMLROOT}/include")
    else()
      message( WARNING "QSMLROOT is not defined" )
      message( FATAL_ERROR "fail to link to a LAPACK library. Try to set HMLP_USE_BLAS=false." )  
    endif()
  else ()
    message( WARNING "compile without BLAS/LAPACK. GOFMM will not be available.")
  endif ()
endif ()



# Vectorization 
# ---------------------------
if ($ENV{HMLP_ARCH_MAJOR} MATCHES "arm")
  #set (HMLP_CFLAGS "-mfpu=neon -mfloat-abi=softfp -march=armv8-a")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -ftree-vectorize")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -mtune=cortex-a57.cortex-a53")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -march=armv8-a+fp+simd")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -mcpu=cortex-a57.cortex-a53")
elseif ($ENV{HMLP_ARCH_MINOR} MATCHES "knl")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -xMIC-AVX512 -DHMLP_MIC_AVX512")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lmemkind")
elseif ($ENV{HMLP_ARCH_MINOR} MATCHES "sandybridge")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -mavx")
elseif ($ENV{HMLP_ARCH_MINOR} MATCHES "haswell")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -mavx")
endif()

# Configure the path structure.
# ---------------------------
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")


# Python support
# ---------------------------
#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})
# message("${PYTHON_LIBRARIES}")
# message("${PYTHON_INCLUDE_DIRS}")
find_program(PYTHON "python")
if (PYTHON)
  set(SETUP_PY_IN ${CMAKE_SOURCE_DIR}/python/setup.in.py)
  set(SETUP_PY    ${CMAKE_BINARY_DIR}/python/setup.py)
  #replace all cmake variables with the correct values
  configure_file(${SETUP_PY_IN} ${SETUP_PY})
  set(HMLP_PYX_IN ${CMAKE_SOURCE_DIR}/python/hmlp.pyx)
  set(HMLP_PYX    ${CMAKE_BINARY_DIR}/python/hmlp.pyx)
  configure_file(${HMLP_PYX_IN} ${HMLP_PYX})
  #set(CHMLP_PXD_IN ${CMAKE_SOURCE_DIR}/python/chmlp.pxd)
  #set(CHMLP_PXD    ${CMAKE_BINARY_DIR}/python/chmlp.pxd)
  #configure_file(${CHMLP_PXD_IN} ${CHMLP_PXD})
  set(GOFMM_PYX_IN ${CMAKE_SOURCE_DIR}/python/gofmm.pyx)
  set(GOFMM_PYX    ${CMAKE_BINARY_DIR}/python/gofmm.pyx)
  configure_file(${GOFMM_PYX_IN} ${GOFMM_PYX})
  #set(CGOFMM_PXD_IN ${CMAKE_SOURCE_DIR}/python/cgofmm.pxd)
  #set(CGOFMM_PXD    ${CMAKE_BINARY_DIR}/python/cgofmm.pxd)
  #configure_file(${CGOFMM_PXD_IN} ${CGOFMM_PXD})
  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_BINARY_DIR})")
  #install(CODE "execute_process(COMMAND CC=${CMAKE_C_COMPILER} ${PYTHON} ${SETUP_PY} build_ext -i)")
endif()


# (OPTIONAL) MPI configurations
# ---------------------------
if ($ENV{HMLP_USE_MPI} MATCHES "true")
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DHMLP_USE_MPI")
endif ()




# Remove -rdynamic
# ---------------------------
set (CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set (CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")




# (OPTIONAL) GPU Compiler sptions
# ---------------------------
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  set (CUDA_TOOLKIT_ROOT_DIR $ENV{HMLP_CUDA_DIR})
  find_package (CUDA 8.0 REQUIRED)
  include (FindCUDA)
  list (APPEND CUDA_NVCC_FLAGS "-std=c++11 -O3")
  #list (APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  list (APPEND CUDA_NVCC_FLAGS --gpu-architecture=sm_60)
  list (APPEND CUDA_NVCC_FLAGS -DHMLP_USE_CUDA)
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DHMLP_USE_CUDA")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L ${CUDA_TOOLKIT_ROOT_DIR}/lib64")
  set (CUDA_PROPAGATE_HOST_FLAGS "false")
  set (CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif ()

if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
  set (HMLP_MAGMA_DIR $ENV{HMLP_MAGMA_DIR})
  list (APPEND CUDA_NVCC_FLAGS -DHMLP_USE_MAGMA)
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DHMLP_USE_MAGMA")
  include_directories ("${HMLP_MAGMA_DIR}/include")
  link_directories ("${HMLP_MAGMA_DIR}/lib")
endif()


# Pthread runtime
# ---------------------------
# set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_PTHREAD_RUNTIME")


# Dump analysis data to google site
# ---------------------------
if ($ENV{HMLP_ANALYSIS_DATA} MATCHES "true")
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DDUMP_ANALYSIS_DATA")
endif()


# Setup all flags
# ---------------------------
set (CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} ${HMLP_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${HMLP_CFLAGS}")


# Linking infos.
# ---------------------------
#if (($ENV{HMLP_USE_BLAS} MATCHES "true") OR ($ENV{HMLP_USE_VML} MATCHES "true"))
#  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
#    include_directories ("${MKL_DIR}/include")
#  else ($ENV{HMLP_USE_INTEL} MATCHES "true")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
#  endif ()
#endif ()

#if (($ENV{HMLP_USE_BLAS} MATCHES "true") AND ($ENV{HMLP_ARCH_MAJOR} MATCHES "arm"))
#  if (DEFINED QSML_DIR)
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${QSML_DIR}/lib -lQSML-0.15.0 -lsymphony-cpu -lsymphonypower")
#    #set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${QSML_DIR}/lib -lQSML-sequential-0.15.0 -lsymphony-cpu -lsymphonypower")
#    include_directories ("${QSML_DIR}/include")
#  endif()
#endif()



# Headers
# ---------------------------
include_directories ("include" "frame" "frame/mpi")
include_directories ("kernel/reference" "kernel/${HMLP_ARCH}")

if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  include_directories ( "${CUDA_TOOLKIT_ROOT_DIR}/include" "${CMAKE_SOURCE_DIR}/kernel/${HMLP_GPU_ARCH}")
endif ()

file (GLOB HMLP_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${HMLP_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
# ---------------------------
file (GLOB FRAME_CC_SRC    ${CMAKE_SOURCE_DIR}/frame/*.c ${CMAKE_SOURCE_DIR}/frame/external/*.c)
file (GLOB FRAME_CXX_SRC   ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB FRAME_CU_SRC    ${CMAKE_SOURCE_DIR}/frame/*.cu)
file (GLOB KERNEL_SRC      ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.cpp)
file (GLOB ASSEMBLY_SRC    ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.s)
file (GLOB PACKAGE_CXX_SRC ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CU_SRC  ${CMAKE_SOURCE_DIR}/package/${HMLP_GPU_ARCH}/*.cu)


#SET(CMAKE_ASM_COMPILE_OBJECT "<CMAKE_ASM_COMPILER> -o <OBJECT> <SOURCE>")


# Build the static library.
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  cuda_add_library (hmlp SHARED ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC}
    ${PACKAGE_CXX_SRC} ${FRAME_CU_SRC} ${PACKAGE_CU_SRC})
else ()
  add_library (hmlp SHARED ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC}
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
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_conv2d.x cudart)
    cuda_add_cublas_to_target(test_conv2d.x)
  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_conv2d.x magma)
  endif()
endif ()


# Build GSKS test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/gsks.cpp)
  add_executable (test_gsks.x ${CMAKE_SOURCE_DIR}/test/test_gsks.cpp)
  target_link_libraries(test_gsks.x hmlp)
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_gsks.x cudart)
    cuda_add_cublas_to_target(test_gsks.x)
  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_gsks.x magma)
  endif()
endif ()

# Build GKMX test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/gkmx.cpp)
  add_executable (test_gkmx.x ${CMAKE_SOURCE_DIR}/test/test_gkmx.cpp)
  target_link_libraries(test_gkmx.x hmlp)
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_gkmx.x cudart)
    cuda_add_cublas_to_target(test_gkmx.x)
  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_gkmx.x magma)
  endif()
endif ()

# Build STRASSEN test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/strassen.cpp)
  add_executable (test_strassen.x ${CMAKE_SOURCE_DIR}/test/test_strassen.cpp)
  target_link_libraries(test_strassen.x hmlp)
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_strassen.x cudart)
    cuda_add_cublas_to_target(test_strassen.x)
  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_strassen.x magma)
  endif()
endif ()

# Build CONV_RELU_POOL test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/conv_relu_pool.cpp)
  add_executable (test_conv_relu_pool.x ${CMAKE_SOURCE_DIR}/test/test_conv_relu_pool.cpp)
  target_link_libraries(test_conv_relu_pool.x hmlp)
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_conv_relu_pool.x cudart)
    cuda_add_cublas_to_target(test_conv_relu_pool.x)
  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_conv_relu_pool.x magma)
  endif()
endif ()

# Build GSKNN test suit
if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/gsknn.cpp)
  add_executable (test_gsknn.x ${CMAKE_SOURCE_DIR}/test/test_gsknn.cpp)
  target_link_libraries(test_gsknn.x hmlp)
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_gsknn.x cudart)
    cuda_add_cublas_to_target(test_gsknn.x)
  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_gsknn.x magma)
  endif()
endif ()

## Build TREE test suit
#if (EXISTS ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/tree.cpp)
#  add_executable (test_tree.x ${CMAKE_SOURCE_DIR}/test/test_tree.cpp)
#  target_link_libraries(test_tree.x hmlp)
#  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
#    target_link_libraries(test_tree.x cudart)
#    cuda_add_cublas_to_target(test_tree.x)
#  endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
#  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
#    target_link_libraries(test_tree.x magma)
#  endif()
#endif ()

# Build SKEL test suit
add_executable (test_skel.x ${CMAKE_SOURCE_DIR}/test/test_skel.cpp)
target_link_libraries(test_skel.x hmlp)
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  target_link_libraries(test_skel.x cudart)
  cuda_add_cublas_to_target(test_skel.x)
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")
if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
  target_link_libraries(test_skel.x magma)
endif()


## build SUPERMATRIX test suit #
add_executable (test_supermatrix.x ${CMAKE_SOURCE_DIR}/test/test_supermatrix.cpp)
target_link_libraries(test_supermatrix.x hmlp)



# Build SPDASKIT test suit
if ($ENV{HMLP_USE_BLAS} MATCHES "true")
  add_executable (test_gofmm.x ${CMAKE_SOURCE_DIR}/test/test_gofmm.cpp)
  target_link_libraries(test_gofmm.x hmlp ${MPI_CXX_LIBRARIES})
  if (MPI_CXX_COMPILE_FLAG)
    set_target_properties(test_gofmm.x PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
  endif()
  if (MPI_LINK_FLAGS)
    set_target_properties(test_gofmm.x PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
  endif()
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries(test_gofmm.x cudart)
    cuda_add_cublas_to_target(test_gofmm.x)
  endif ()
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries(test_gofmm.x magma)
  endif()
else ()
  message( WARNING "GOFMM is not compiled becuase HMLP_USE_BLAS=false" )
endif ($ENV{HMLP_USE_BLAS} MATCHES "true")


# Also build 
set(ARTIFACT_PATH $ENV{HMLP_ARTIFACT_PATH})
if (ARTIFACT_PATH MATCHES "sc17gofmm")
  set(ARTIFACT_PATH "${CMAKE_SOURCE_DIR}/artifact/${ARTIFACT_PATH}")
  file (GLOB ARTIFACT_CXX_SRC    ${ARTIFACT_PATH}/*.cpp)
  file (GLOB ARTIFACT_SCRIPTS    ${ARTIFACT_PATH}/*.sh)
  foreach( artifactsrc ${ARTIFACT_CXX_SRC} )
    string( REPLACE ".cpp" ".x" artifactexe ${artifactsrc} )
    get_filename_component( artifactexe ${artifactsrc} NAME_WE )
    add_executable( "${artifactexe}.x" ${artifactsrc} )
    target_link_libraries( "${artifactexe}.x" hmlp ${MPI_CXX_LIBRARIES} )
  endforeach( artifactsrc ${ARTIFACT_CXX_SRC} )
  install (FILES ${ARTIFACT_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()


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
  # add_executable (test_spdaskit_gpu.x ${CMAKE_SOURCE_DIR}/test/test_spdaskit_gpu.cu)
  # target_link_libraries(test_spdaskit_gpu.x hmlp)
  # cuda_add_cublas_to_target(test_spdaskit_gpu.x)
endif ()
if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
  target_link_libraries(test_gkmm_gpu.x magma)
  target_link_libraries(test_strassen_gpu.x magma)
endif()






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
message ("MPICXX       =${MPI_CXX_COMPILER}")
message ("MPICXXFLAGS  =${MPI_CXX_COMPILE_FLAGS}")
message ("MPICXXINCLUDE=${MPI_CXX_INCLUDE_PATH}")
message ("MPICXX Linker=${MPI_CXX_LINK_FLAGS}")
message ("MPICXXLIBS   =${MPI_CXX_LIBRARIES}")
message ("AR           =${CMAKE_AR}")
message ("RANLIB       =${CMAKE_RANLIB}")
message ("===================================================")
message ("HMLP_GPU_ARCH=${HMLP_GPU_ARCH}")
message ("HOST_COMPILER=${CUDA_HOST_COMPILER}")
message ("NVCC_VERSION =${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
message ("NVCCFLAGS    =${CUDA_NVCC_FLAGS}")
message ("===================================================")
message ("MKLROOT      =$ENV{MKLROOT}")
message ("OPENBLASROOT =$ENV{OPENBLASROOT}")
message ("QSMLROOT     =$ENV{QSMLROOT}")
message ("BLAS_LIBRARY =${BLAS_LIBRARIES}")
message ("BLAS_LINKER  =${BLAS_LINKER_FLAGS}")
message ("===================================================")
