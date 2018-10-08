# ---[ Check cmake version.
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

# ---[ Project specification.
project(HMLP C CXX)
set(HMLP_VERSION_MAJOR 1)
set(HMLP_VERSION_MINOR 1)
set(HMLP_VERSION_PATCH 0)
set(HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})

# ---[ Build options (not completed yet).
set(HMLP_LIBRARY_TYPE "default" CACHE STRING "Type of hmlp library (shared, static, or default) to build")
set_property(CACHE HMLP_LIBRARY_TYPE PROPERTY STRINGS default static shared)
option(HMLP_BUILD_TOOLS "Build command-line tools" ON)
option(HMLP_BUILD_UNIT_TESTS "Build hmlp unit tests" ON)
option(HMLP_BUILD_MOCK_TESTS "Build hmlp mock tests" ON)
option(HMLP_BUILD_BENCHMARKS "Build hmlp micro-benchmarks" ON)
option(HMLP_BUILD_ARTIFACTS "Build SC'XX artifacts" ON)

# --[ Whether or not enable testing?
if(HMLP_BUILD_UNIT_TESTS OR HMLP_BUILD_MOCK_TESTS)
  ENABLE_TESTING()
endif()

# ---[ Decide architecture.
set(HMLP_SUPPORTED_PLATFORM TRUE)
if(NOT CMAKE_SYSTEM_PROCESSOR)
  message(WARNING "Target processor architecture is not specified.")
  set(HMLP_SUPPORTED_PLATFORM FALSE)
elseif(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|armv7|armv7f|armv7s|armv7k|arm64|aarch64)$")
  message(WARNING "Target processor architecture ${CMAKE_SYSTEM_PROCESSOR} is not supported in hmlp.")
  set(HMLP_SUPPORTED_PLATFORM FALSE)
endif()


message(STATUS "${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_SYSTEM_NAME}")
set(HMLP_ARCH_MAJOR     $ENV{HMLP_ARCH_MAJOR})
set(HMLP_ARCH_MINOR     $ENV{HMLP_ARCH_MINOR})
set(HMLP_ARCH           ${HMLP_ARCH_MAJOR}/${HMLP_ARCH_MINOR})
set(HMLP_GPU_ARCH_MAJOR $ENV{HMLP_GPU_ARCH_MAJOR})
set(HMLP_GPU_ARCH_MINOR $ENV{HMLP_GPU_ARCH_MINOR})
set(HMLP_GPU_ARCH       ${HMLP_GPU_ARCH_MAJOR}/${HMLP_GPU_ARCH_MINOR})

# Detecting MacOS.
# ---------------------------
if ( APPLE )
  message( STATUS "MacOS detected")
  set (CMAKE_MACOSX_RPATH TRUE)
endif ()


# ---[ Download dependencies.
set(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps
  CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps
  CACHE PATH "Confu-style dependencies binary directory")

# ---[ Download googletest.
if(HMLP_BUILD_MOCK_TESTS OR HMLP_BUILD_UNIT_TESTS)
  if(HMLP_SUPPORTED_PLATFORM AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
    message(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
    configure_file(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    set(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
  ENDIF()
ENDIF()

# ---[ Download googlebenchmark.
if(HMLP_BUILD_BENCHMARKS)
  if(HMLP_SUPPORTED_PLATFORM AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
    message(STATUS "Downloading Google Benchmark to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
    configure_file(cmake/DownloadGoogleBenchmark.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
    execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
    execute_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
    set(GOOGLEBENCHMARK_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark" CACHE STRING "Google Benchmark source directory")
  endif()
endif()

# ---[ Compiler Options (HMLP can use Intel or GNU compilers.)
if ("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")

  # using AppleClang
  message( STATUS "Clang compiler." )
  message( WARNING "You may need to install OpenMP. [e.g. brew install libomp]" ) 
  #set (HMLP_CFLAGS            "${HMLP_CFLAGS} -g -O3 -Xpreprocessor -fopenmp -lomp")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -g -O3 -Xpreprocessor -fopenmp -L/usr/local/opt/libomp/lib -lomp")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -fPIC -D_POSIX_C_SOURCE=200112L")
  set (HMLP_PYTHON_CFLAGS     "-g -O3 -fopenmp -arch x86_64 -m64")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")

  # using GCC
  message( STATUS "GNU compiler." ) 
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -g -O3 -fopenmp")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -fPIC -D_POSIX_C_SOURCE=200112L")
  set (HMLP_PYTHON_CFLAGS     "-g -O3 -fopenmp -arch x86_64 -m64")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")

elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")

  # using Intel C++
  message( STATUS "Intel compiler." )
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -g -O3 -qopenmp")
  #set (HMLP_CFLAGS            "${HMLP_CFLAGS} -Werror -Wall -pedantic -fPIC")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -Wall -pedantic -fPIC -D_POSIX_C_SOURCE=200112L")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -DUSE_INTEL")
  set (HMLP_PYTHON_CFLAGS     "-g -std=c++11 -O3 -qopenmp -arch x86_64 -m64")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp")

endif()

# Software dependencies.
# ---------------------------
set (MKLROOT      $ENV{MKLROOT})
set (BLISROOT     $ENV{BLISROOT})
set (OPENBLASROOT $ENV{OPENBLASROOT})
set (QSMLROOT     $ENV{QSMLROOT})


# Try to detect and link BLAS/LAPACK
# ---------------------------
if ($ENV{HMLP_USE_BLAS} MATCHES "true")

  # HMLP use flag USE_BLAS for preprocessing.
  set (HMLP_CFLAGS "${HMLP_CFLAGS} -DUSE_BLAS")

  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    # Use MKL by default.
    message( STATUS "Use MKL for BLAS/LAPACK" )
    set (HMLP_CFLAGS "${HMLP_CFLAGS} -DUSE_VML -mkl=parallel")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
    include_directories ("${MKLROOT}/include")
  else ()

    # Look for OpenBLAS or built-in BLAS/LAPACK libraries.
    if ( OPENBLASROOT )
      message( STATUS "Use OpenBLAS for BLAS/LAPACK" )
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${OPENBLASROOT} -lopenblas")
    else ()
      # Check if built-in BLAS is available.
      message( STATUS "Use built-in BLAS/LAPACK from /usr/lib" )
      set (HMLP_CFLAGS "${HMLP_CFLAGS} -lblas -llapack")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lblas -llapack")
      # set(BLAS_DIR "/usr/lib")
      # set(LAPACK_DIR "/usr/lib")
      # find_package(BLAS REQUIRED)
      # find_package(LAPACK REQUIRED)
      # if ( BLAS_LINKER_FLAGS )
      #   message( "${BLAS_LINKER_FLAGS}" )
      #   message( "${BLAS_LIBRARIES}" )
      #   set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
      # else ()
      #   message( WARNING "cmake FindBLAS fails to generate a proper linker for BLAS" )
      #   message( FATAL_ERROR "fail to link to a BLAS library. Try to set HMLP_USE_BLAS=false." )  
      # endif ()
      # # Check if built-in LAPACK is available.
      # if ( LAPACK_LINKER_FLAGS )
      #   message("${LAPACK_LINKER_FLAGS}")
      #   message("${LAPACK_LIBRARIES}")
      #   set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
      # else ()
      #   message( WARNING "cmake FindBLAS fails to generate a proper linker for LAPACK" )
      #   message( FATAL_ERROR "fail to link to a LAPACK library. Try to set HMLP_USE_BLAS=false." )  
      # endif ()
    endif()
  endif ()
     
else ()
  # HMLP will still compile, but the behavior of GOFMM is undefined.
  message( WARNING "compile without BLAS/LAPACK. GOFMM will not be available.")
endif ()





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



# ---[ Vectorization 
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
elseif ($ENV{HMLP_ARCH_MINOR} MATCHES "skx")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -xCORE-AVX2 -axCORE-AVX512,MIC-AVX512")
endif()


# ---[ Whether to compile the runtime system
if ($ENV{HMLP_HAVE_RUNTIME} MATCHES "true")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -DHMLP_HAVE_RUNTIME")
endif()



# ---[ Configure the path structure.
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")


# ---[ Python support
find_program(VIRTUALENV "virtualenv")


if (VIRTUALENV)
	set(INSTA_SH_IN ${CMAKE_SOURCE_DIR}/python/install.in.sh)
	set(INSTA_SH    ${CMAKE_BINARY_DIR}/python/install.sh)
	configure_file(${INSTA_SH_IN} ${INSTA_SH})
  set(SETUP_PY_IN ${CMAKE_SOURCE_DIR}/python/setup.in.py)
  set(SETUP_PY    ${CMAKE_BINARY_DIR}/python/setup.py)
  configure_file(${SETUP_PY_IN} ${SETUP_PY})
  set(HMLP_PYX_IN ${CMAKE_SOURCE_DIR}/python/hmlp.py)
  set(HMLP_PYX    ${CMAKE_BINARY_DIR}/python/hmlp.pyx)
  configure_file(${HMLP_PYX_IN} ${HMLP_PYX})
  set(GOFMM_PYX_IN ${CMAKE_SOURCE_DIR}/python/gofmm.py)
  set(GOFMM_PYX    ${CMAKE_BINARY_DIR}/python/gofmm.pyx)
  configure_file(${GOFMM_PYX_IN} ${GOFMM_PYX})
	## Create a Virtualenv environment for Python 2.7. 
	install(CODE "execute_process(COMMAND virtualenv --system-site-packages ${CMAKE_BINARY_DIR}/python)")
	## Jump to the virtual environment.
	#install(CODE "execute_process(COMMAND cd ${CMAKE_BINARY_DIR}/python)")
	## Active the virtual environment (bash shell).
	#install(CODE "execute_process(COMMAND source ./bin/activate)")
	#install(CODE "execute_process(COMMAND source ${CMAKE_BINARY_DIR}/python/bin/activate)")
	## Ensure pip ≥8.1 is installed.
	#install(CODE "execute_process(COMMAND easy_install -U pip)")
	## Install Cython.
	#install(CODE "execute_process(COMMAND pip install Cython)")
	
	

	#install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --prefix=${CMAKE_BINARY_DIR})")
  #install(CODE "execute_process(COMMAND CC=${CMAKE_C_COMPILER} ${PYTHON} ${SETUP_PY} build_ext -i)")
endif()





# ---[ (OPTIONAL) MPI configurations
if ($ENV{HMLP_USE_MPI} MATCHES "true")
  message("MPI is enable") 
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(HMLP_CFLAGS          "${HMLP_CFLAGS} -DHMLP_USE_MPI")
else ()
  message("MPI is disable") 
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
  list (APPEND CUDA_NVCC_FLAGS -gencode arch=compute_35,code=sm_35)
  list (APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
  #list (APPEND CUDA_NVCC_FLAGS --gpu-architecture=sm_60)
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


# Dump analysis data to google site
# ---------------------------
if ($ENV{HMLP_ANALYSIS_DATA} MATCHES "true")
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DDUMP_ANALYSIS_DATA")
endif()


# Setup all flags
# ---------------------------
set (CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} ${HMLP_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${HMLP_CFLAGS} -std=c++11")

# Headers
# ---------------------------
include_directories ("include" "frame" "gofmm")
include_directories ("frame/base" "frame/containers" "frame/mpi")
include_directories ("kernel/reference" "kernel/${HMLP_ARCH}")

if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  include_directories ( "${CUDA_TOOLKIT_ROOT_DIR}/include" "${CMAKE_SOURCE_DIR}/kernel/${HMLP_GPU_ARCH}")
endif ()

file (GLOB HMLP_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${HMLP_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
# ---------------------------
file (GLOB FRAME_CC_SRC      ${CMAKE_SOURCE_DIR}/frame/external/*.c)
file (GLOB FRAME_CXX_SRC     ${CMAKE_SOURCE_DIR}/frame/base/*.cpp)
file (GLOB FRAME_MPI_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/mpi/*.cpp)
file (GLOB FRAME_CU_SRC      ${CMAKE_SOURCE_DIR}/frame/*.cu)
file (GLOB GOFMM_CU_SRC      ${CMAKE_SOURCE_DIR}/frame/gofmm/*.cu)
file (GLOB KERNEL_SRC        ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.cpp)
file (GLOB ASSEMBLY_SRC      ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.s)
file (GLOB PACKAGE_CXX_SRC   ${CMAKE_SOURCE_DIR}/package/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CU_SRC    ${CMAKE_SOURCE_DIR}/package/${HMLP_GPU_ARCH}/*.cu)


# Build the shared library.
# ---------------------------
if ($ENV{HMLP_USE_CUDA} MATCHES "true")
  cuda_add_library (hmlp SHARED ${FRAME_CC_SRC} 
    ${FRAME_CXX_SRC} ${FRAME_MPI_CXX_SRC} ${KERNEL_SRC}
    ${PACKAGE_CXX_SRC} ${FRAME_CU_SRC} ${GOFMM_CU_SRC} ${PACKAGE_CU_SRC})
else ()
  add_library (hmlp SHARED ${FRAME_CC_SRC} 
    ${FRAME_CXX_SRC} ${FRAME_MPI_CXX_SRC} ${KERNEL_SRC}
    ${PACKAGE_CXX_SRC} ${ASSEMBLY_SRC})
endif ($ENV{HMLP_USE_CUDA} MATCHES "true")


# # Build HMLP executables.
# # ---------------------------
# file (GLOB TEST_CXX_SRC    ${CMAKE_SOURCE_DIR}/test/*.cpp)
# foreach( testsrc ${TEST_CXX_SRC} )
#   string( REPLACE ".cpp" ".x" testexe ${testsrc} )
#   get_filename_component( testexe ${testsrc} NAME_WE )
#   add_executable( "${testexe}.x" ${testsrc} )
#   target_link_libraries( "${testexe}.x" hmlp ${MPI_CXX_LIBRARIES} )
#   set_target_properties( "${testexe}.x" PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
#   set_target_properties( "${testexe}.x" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#   if ($ENV{HMLP_USE_CUDA} MATCHES "true")
#     target_link_libraries( "${testexe}.x" cudart )
#     cuda_add_cublas_to_target( "${testexe}.x" )
#   endif ()
#   if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
#     target_link_libraries( "${testexe}.x" magma )
#   endif()
# endforeach()


# Build GOFMM executables.
# ---------------------------
file (GLOB TEST_CXX_SRC    ${CMAKE_SOURCE_DIR}/gofmm/test/*.cpp)
foreach( testsrc ${TEST_CXX_SRC} )
  string( REPLACE ".cpp" ".x" testexe ${testsrc} )
  get_filename_component( testexe ${testsrc} NAME_WE )
  add_executable( "${testexe}.x" ${testsrc} )
  target_link_libraries( "${testexe}.x" hmlp ${MPI_CXX_LIBRARIES} )
  set_target_properties( "${testexe}.x" PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
  set_target_properties( "${testexe}.x" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
  if ($ENV{HMLP_USE_CUDA} MATCHES "true")
    target_link_libraries( "${testexe}.x" cudart )
    cuda_add_cublas_to_target( "${testexe}.x" )
  endif ()
  if ($ENV{HMLP_USE_MAGMA} MATCHES "true")
    target_link_libraries( "${testexe}.x" magma )
  endif()
endforeach()

# Install GOFMM shell script
# ---------------------------
file (GLOB GOFMM_SCRIPTS ${CMAKE_SOURCE_DIR}/gofmm/test/*.sh)
install (FILES ${GOFMM_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)



## Build paper artifacts
# ---------------------------
set(ARTIFACT_PATH $ENV{HMLP_ARTIFACT_PATH})
if (ARTIFACT_PATH)
  set(ARTIFACT_PATH "${CMAKE_SOURCE_DIR}/artifact/${ARTIFACT_PATH}")
  file (GLOB ARTIFACT_CXX_SRC    ${ARTIFACT_PATH}/*.cpp)
  file (GLOB ARTIFACT_SCRIPTS    ${ARTIFACT_PATH}/*.sh)
  file (GLOB ARTIFACT_DATASETS   ${ARTIFACT_PATH}/datasets/*.m)
  foreach( artifactsrc ${ARTIFACT_CXX_SRC} )
    string( REPLACE ".cpp" ".x" artifactexe ${artifactsrc} )
    get_filename_component( artifactexe ${artifactsrc} NAME_WE )
    add_executable( "${artifactexe}.x" ${artifactsrc} )
    target_link_libraries( "${artifactexe}.x" hmlp ${MPI_CXX_LIBRARIES} )
    if ($ENV{HMLP_USE_CUDA} MATCHES "true")
      target_link_libraries( "${artifactexe}.x" cudart )
      cuda_add_cublas_to_target( "${artifactexe}.x" )
    endif()
  endforeach( artifactsrc ${ARTIFACT_CXX_SRC} )
  install (FILES ${ARTIFACT_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/bin)
  install (FILES ${ARTIFACT_DATASETS} DESTINATION ${CMAKE_BINARY_DIR}/bin/datasets)
endif()


# ## Build the sandbox
# # ---------------------------
# set(BUILD_SANDBOX $ENV{HMLP_BUILD_SANDBOX})
# if (BUILD_SANDBOX MATCHES "true")
#   file (GLOB SANDBOX_ENV ${CMAKE_SOURCE_DIR}/set_env.sh )
# 
#   file (GLOB SANDBOX_SRC ${CMAKE_SOURCE_DIR}/frame/blas_lapack_prototypes.hpp
#                          ${CMAKE_SOURCE_DIR}/frame/hmlp_blas_lapack.cpp
#                          ${CMAKE_SOURCE_DIR}/frame/hmlp_blas_lapack.cu
#                          ${CMAKE_SOURCE_DIR}/frame/hmlp_blas_lapack.h
#                          ${CMAKE_SOURCE_DIR}/frame/hmlp_thread.cpp
#                          ${CMAKE_SOURCE_DIR}/frame/hmlp_thread.hpp
#                          #${CMAKE_SOURCE_DIR}/frame/hmlp_runtime.cpp
#                          #${CMAKE_SOURCE_DIR}/frame/hmlp_runtime.hpp
#                          #${CMAKE_SOURCE_DIR}/frame/hmlp_device.cpp
#                          #${CMAKE_SOURCE_DIR}/frame/hmlp_device.hpp
#                          ${CMAKE_SOURCE_DIR}/frame/hmlp_util.hpp )
# 
#                        #file (GLOB SANDBOX_EXT ${CMAKE_SOURCE_DIR}/frame/external/*)
#   file (GLOB SANDBOX_PRI ${CMAKE_SOURCE_DIR}/frame/primitives/gnbx.hpp)
#   file (GLOB SANDBOX_PKG ${CMAKE_SOURCE_DIR}/package/x86_64/sandybridge/gnbx.cpp)
#   file (GLOB SANDBOX_EXE ${CMAKE_SOURCE_DIR}/test/test_gnbx.cpp)
# 
#   #file (GLOB SANDBOX_INC ${CMAKE_SOURCE_DIR}/include/*)
#   install(FILES ${SANDBOX_ENV} DESTINATION ${CMAKE_BINARY_DIR}/sandbox)
#   install(FILES ${CMAKE_SOURCE_DIR}/cmake/sandbox_CMakeLists.txt DESTINATION 
#     ${CMAKE_BINARY_DIR}/sandbox RENAME CMakeLists.txt)
#   install(FILES ${SANDBOX_SRC} DESTINATION ${CMAKE_BINARY_DIR}/sandbox/frame)
#   #install(FILES ${SANDBOX_EXT} DESTINATION ${CMAKE_BINARY_DIR}/sandbox/frame/external)
#   install(FILES ${SANDBOX_PRI} DESTINATION ${CMAKE_BINARY_DIR}/sandbox/frame/primitives)
#   install(FILES ${SANDBOX_PKG} DESTINATION ${CMAKE_BINARY_DIR}/sandbox/package/x86_64/sandybridge)
#   install(FILES ${SANDBOX_EXE} DESTINATION ${CMAKE_BINARY_DIR}/sandbox/test)
#   install(DIRECTORY ${CMAKE_SOURCE_DIR}/frame/external DESTINATION ${CMAKE_BINARY_DIR}/sandbox/frame)
#   install(DIRECTORY ${CMAKE_SOURCE_DIR}/kernel DESTINATION ${CMAKE_BINARY_DIR}/sandbox)
#   install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION ${CMAKE_BINARY_DIR}/sandbox)
# endif()


# ---[ Report.
message("===================================================")
message("Version      =${HMLP_VERSION}")
message("Source       =${CMAKE_SOURCE_DIR}")
message("Target       =${CMAKE_BINARY_DIR}")
message("Platform     =${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_SYSTEM_NAME}")
message("HMLP_ARCH    =${HMLP_ARCH}")
message("CC           =${CMAKE_C_COMPILER}")
message("CFLAGS       =${CMAKE_C_FLAGS}")
message("C Linker     =${CMAKE_C_LINK_FLAGS}")
message("CXX          =${CMAKE_CXX_COMPILER}")
message("CXXFLAGS     =${CMAKE_CXX_FLAGS}")
message("CXX Linker   =${CMAKE_CXX_LINK_FLAGS}")
message("EXE Linker   =${CMAKE_EXE_LINKER_FLAGS}")
message("MPICXX       =${MPI_CXX_COMPILER}")
message("MPICXXFLAGS  =${MPI_CXX_COMPILE_FLAGS}")
message("MPICXXINCLUDE=${MPI_CXX_INCLUDE_PATH}")
message("MPICXX Linker=${MPI_CXX_LINK_FLAGS}")
message("MPICXXLIBS   =${MPI_CXX_LIBRARIES}")
message("AR           =${CMAKE_AR}")
message("RANLIB       =${CMAKE_RANLIB}")
message("===================================================")
message("HMLP_GPU_ARCH=${HMLP_GPU_ARCH}")
message("HOST_COMPILER=${CUDA_HOST_COMPILER}")
message("NVCC_VERSION =${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
message("NVCCFLAGS    =${CUDA_NVCC_FLAGS}")
message("===================================================")
message("MKLROOT      =$ENV{MKLROOT}")
message("BLISROOT     =$ENV{BLISROOT}")
message("OPENBLASROOT =$ENV{OPENBLASROOT}")
message("QSMLROOT     =$ENV{QSMLROOT}")
message("BLAS_LIBRARY =${BLAS_LIBRARIES}")
message("BLAS_LINKER  =${BLAS_LINKER_FLAGS}")
message("===================================================")
