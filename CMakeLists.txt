# ---[ Check cmake version.
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

# ---[ Project specification.
PROJECT(hmlp C CXX)
SET(HMLP_VERSION_MAJOR 1)
SET(HMLP_VERSION_MINOR 1)
SET(HMLP_VERSION_PATCH 0)
SET(HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})


IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

SET(CMAKE_RELEASE_POSTFIX "")
SET(CMAKE_DEBUG_POSTFIX "-debug")
# ---[ Download dependencies.
SET(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps CACHE PATH "Confu-style dependencies source directory")
SET(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps CACHE PATH "Confu-style dependencies binary directory")

# ---[ Build options (not completed yet).
SET(LIBRARY_TYPE "default" CACHE STRING "Type of hmlp library (shared, static, or default) to build")
SET_PROPERTY(CACHE LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(BUILD_TOOLS "Build command-line tools" ON)
OPTION(BUILD_UNIT_TESTS "Build hmlp unit tests" ON)
OPTION(BUILD_MOCK_TESTS "Build hmlp mock tests" ON)
OPTION(BUILD_BENCHMARKS "Build hmlp micro-benchmarks" ON)
OPTION(BUILD_ARTIFACTS "Build SC'XX artifacts" ON)

# --[ Whether or not enable testing?
IF(BUILD_UNIT_TESTS OR BUILD_MOCK_TESTS)
  ENABLE_TESTING()
ENDIF()

# ---[ Decide architecture.
SET(SUPPORTED_PLATFORM TRUE)
IF(NOT CMAKE_SYSTEM_PROCESSOR)
  MESSAGE(WARNING "Target processor architecture is not specified.")
  SET(SUPPORTED_PLATFORM FALSE)
ELSEIF(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|armv7|armv7f|armv7s|armv7k|arm64|aarch64)$")
  MESSAGE(WARNING "Target processor architecture ${CMAKE_SYSTEM_PROCESSOR} is not supported in hmlp.")
  SET(SUPPORTED_PLATFORM FALSE)
ENDIF()

MESSAGE(STATUS "${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_SYSTEM_NAME}")
SET(HMLP_ARCH_MAJOR     $ENV{HMLP_ARCH_MAJOR})
SET(HMLP_ARCH_MINOR     $ENV{HMLP_ARCH_MINOR})
SET(HMLP_ARCH           ${HMLP_ARCH_MAJOR}/${HMLP_ARCH_MINOR})
SET(HMLP_GPU_ARCH_MAJOR $ENV{HMLP_GPU_ARCH_MAJOR})
SET(HMLP_GPU_ARCH_MINOR $ENV{HMLP_GPU_ARCH_MINOR})
SET(HMLP_GPU_ARCH       ${HMLP_GPU_ARCH_MAJOR}/${HMLP_GPU_ARCH_MINOR})

# Detecting MacOS.
# ---------------------------
#IF( APPLE )
#  message( STATUS "MacOS detected")
#  set (CMAKE_MACOSX_RPATH TRUE)
#ENDIF()
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# ---[ Download googletest.
IF(BUILD_MOCK_TESTS OR BUILD_UNIT_TESTS)
  IF(SUPPORTED_PLATFORM AND NOT DEFINED GOOGLETEST_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    SET(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
  ENDIF()
ENDIF()

# ---[ Download googlebenchmark.
IF(BUILD_BENCHMARKS)
  IF(SUPPORTED_PLATFORM AND NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Benchmark to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googlebenchmark-download")
    SET(GOOGLEBENCHMARK_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googlebenchmark" CACHE STRING "Google Benchmark source directory")
  ENDIF()
ENDIF()

# Compiler Options (HMLP can use Intel or GNU compilers.)
# ---------------------------
IF("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  # ---[ CLANG/CLANG++
  MESSAGE(STATUS "Clang compiler.")
  MESSAGE(WARNING "You may need to install OpenMP. [e.g. brew install libomp]") 
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -O3 -Xpreprocessor -fopenmp -L/usr/local/opt/libomp/lib -lomp")
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -fPIC -D_POSIX_C_SOURCE=200112L")
  SET(HMLP_PYTHON_CFLAGS "-O3 -fopenmp -arch x86_64 -m64")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -fopenmp -lm")
ELSEIF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  # ---[ GCC/G++
  MESSAGE(STATUS "GNU compiler.") 
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -O3 -fopenmp -m64")
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -fPIC -D_POSIX_C_SOURCE=200112L")
  # ---[ Line and branch coverage analysis.
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -fprofile-arcs -ftest-coverage -fPIC")
  #SET(HMLP_CFLAGS "${HMLP_CFLAGS} -coverage -fPIC")
  SET(HMLP_PYTHON_CFLAGS "-O3 -fopenmp -arch x86_64 -m64")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -fopenmp -lm")
ELSEIF("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
  # ---[ ICC/ICPC
  MESSAGE(STATUS "Intel compiler.")
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -O3 -qopenmp -m64")
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -Wall -pedantic -fPIC -D_POSIX_C_SOURCE=200112L")
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -DUSE_INTEL")
  SET(HMLP_PYTHON_CFLAGS     "-std=c++11 -O3 -qopenmp -arch x86_64 -m64")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread -qopenmp -lm")
ELSE()
  MESSAGE(FATAL_ERROR "Unsupported compiler type ${CMAKE_C_COMPILER_ID}")
ENDIF()

# Software dependencies.
# ---------------------------
SET(MKLROOT      $ENV{MKLROOT})
SET(BLISROOT     $ENV{BLISROOT})
SET(OPENBLASROOT $ENV{OPENBLASROOT})
SET(NETLIBROOT   $ENV{NETLIBROOT})
SET(QSMLROOT     $ENV{QSMLROOT})

# Try to detect and link BLAS/LAPACK
# ---------------------------
if ($ENV{HMLP_USE_BLAS} MATCHES "true")
  # ---[ HMLP use flag USE_BLAS for preprocessing.
  SET(HMLP_CFLAGS "${HMLP_CFLAGS} -DUSE_BLAS")
  IF("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    # ---[ Use MKL by default.
    MESSAGE(STATUS "Use MKL for BLAS/LAPACK")
    SET(HMLP_CFLAGS "${HMLP_CFLAGS} -DUSE_VML -mkl=parallel")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
  ELSE()
    # Look for OpenBLAS or built-in BLAS/LAPACK libraries.
    IF(OPENBLASROOT)
      MESSAGE(STATUS "Use OpenBLAS for BLAS/LAPACK")
      SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${OPENBLASROOT} -lopenblas")
    ELSEIF(NETLIBROOT)
      MESSAGE(STATUS "Use NETLIB BLAS/LAPACK")
      FIND_PACKAGE(BLAS REQUIRED)
      FIND_PACKAGE(LAPACK REQUIRED)
      #SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${NETLIBROOT} -lblas -llapack")
      SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    ELSE()
      MESSAGE(FATAL_ERROR "Fail to link to BLAS/LAPACK library. Try to set HMLP_USE_BLAS=false.")  
    ENDIF()
  ENDIF()     
ELSE()
  # ---[ HMLP will still compile, but the behavior of GOFMM is undefined.
  MESSAGE(WARNING "Compile without BLAS/LAPACK. GOFMM will not be available.")
ENDIF()


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
  #set (HMLP_CFLAGS            "${HMLP_CFLAGS} -xCORE-AVX2 -axCORE-AVX512,MIC-AVX512")
  set (HMLP_CFLAGS            "${HMLP_CFLAGS} -march=skylake -mavx -mavx2 -mavx512f")
endif()


# ---[ Configure the path structure.
#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set (CMAKE_RELEASE_POSTFIX "")
#set (CMAKE_DEBUG_POSTFIX "-debug")


# ---[ Python support
find_program(VIRTUALENV "virtualenv")


if (VIRTUALENV)
	set(INSTA_SH_IN ${hmlp_SOURCE_DIR}/python/install.in.sh)
	set(INSTA_SH    ${CMAKE_BINARY_DIR}/python/install.sh)
	configure_file(${INSTA_SH_IN} ${INSTA_SH})
  set(SETUP_PY_IN ${hmlp_SOURCE_DIR}/python/setup.in.py)
  set(SETUP_PY    ${CMAKE_BINARY_DIR}/python/setup.py)
  configure_file(${SETUP_PY_IN} ${SETUP_PY})
  set(HMLP_PYX_IN ${hmlp_SOURCE_DIR}/python/hmlp.py)
  set(HMLP_PYX    ${CMAKE_BINARY_DIR}/python/hmlp.pyx)
  configure_file(${HMLP_PYX_IN} ${HMLP_PYX})
  set(GOFMM_PYX_IN ${hmlp_SOURCE_DIR}/python/gofmm.py)
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
#message(${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS})
#message(${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS})
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
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${HMLP_CFLAGS}")
SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HMLP_CFLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")






# Headers
# ---------------------------
#include_directories ("include" "frame" "gofmm")
#include_directories ("frame/base" "frame/containers" "frame/mpi")
#include_directories ("kernel/reference" "kernel/${HMLP_ARCH}")

#if ($ENV{HMLP_USE_CUDA} MATCHES "true")
#  include_directories ( "${CUDA_TOOLKIT_ROOT_DIR}/include" "${hmlp_SOURCE_DIR}/kernel/${HMLP_GPU_ARCH}")
#endif ()

#file (GLOB HMLP_HDRS ${hmlp_SOURCE_DIR}/include/*)
#install (FILES ${HMLP_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
# ---------------------------
file (GLOB FRAME_CC_SRC      ${hmlp_SOURCE_DIR}/frame/external/*.c)
file (GLOB FRAME_CXX_SRC     ${hmlp_SOURCE_DIR}/frame/base/*.cpp)
file (GLOB FRAME_MPI_CXX_SRC ${hmlp_SOURCE_DIR}/frame/mpi/*.cpp)
file (GLOB FRAME_CU_SRC      ${hmlp_SOURCE_DIR}/frame/*.cu)
file (GLOB GOFMM_CU_SRC      ${hmlp_SOURCE_DIR}/frame/gofmm/*.cu)
file (GLOB KERNEL_SRC        ${hmlp_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CXX_SRC   ${hmlp_SOURCE_DIR}/package/${HMLP_ARCH}/*.cpp)
file (GLOB PACKAGE_CU_SRC    ${hmlp_SOURCE_DIR}/package/${HMLP_GPU_ARCH}/*.cu)

FILE(GLOB CC_SRC 
  frame/external/*.c
  frame/base/*.cpp
  frame/mpi/*.cpp
  kernel/${HMLP_ARCH}/*.cpp
  package/${HMLP_ARCH}/*.cpp)

FILE(GLOB CU_SRC 
  frame/*.cu
  frame/gofmm/*.cu
  package/${HMLP_GPU_ARCH}/*.cu)

# Public Headers
# ---------------------------
LIST(APPEND INC include)




## Build the shared library.
## ---------------------------
#if ($ENV{HMLP_USE_CUDA} MATCHES "true")
#  cuda_add_library (hmlp SHARED 
#else ()
#  add_library (hmlp SHARED ${FRAME_CC_SRC} 
#    ${FRAME_CXX_SRC} ${FRAME_MPI_CXX_SRC} ${KERNEL_SRC}
#    ${PACKAGE_CXX_SRC})
#endif ($ENV{HMLP_USE_CUDA} MATCHES "true")


IF($ENV{HMLP_USE_CUDA} MATCHES "true")
  IF(LIBRARY_TYPE STREQUAL "default")
    CUDA_ADD_LIBRARY(hmlp ${CU_SRC})
  ELSEIF(LIBRARY_TYPE STREQUAL "shared")
    CUDA_ADD_LIBRARY(hmlp SHARED ${CU_SRC})
  ELSEIF(LIBRARY_TYPE STREQUAL "static")
    CUDA_ADD_LIBRARY(hmlp STATIC ${CU_SRC})
  ELSE()
    MESSAGE(FATAL_ERROR "Unsupported library type ${LIBRARY_TYPE}")
  ENDIF()
  TARGET_INCLUDE_DIRECTORIES(cuel BEFORE PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")
ELSE()
  IF(LIBRARY_TYPE STREQUAL "default")
    ADD_LIBRARY(hmlp ${CC_SRC})
  ELSEIF(LIBRARY_TYPE STREQUAL "shared")
    ADD_LIBRARY(hmlp SHARED ${CC_SRC})
  ELSEIF(LIBRARY_TYPE STREQUAL "static")
    ADD_LIBRARY(hmlp STATIC ${CC_SRC})
  ELSE()
    MESSAGE(FATAL_ERROR "Unsupported library type ${LIBRARY_TYPE}")
  ENDIF()
ENDIF()
SET_TARGET_PROPERTIES(hmlp PROPERTIES PUBLIC_HEADER include/hmlp.h)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PUBLIC include)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE frame)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE gofmm)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE frame/base)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE frame/containers)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE frame/primitives)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE kernel/reference)
TARGET_INCLUDE_DIRECTORIES(hmlp BEFORE PRIVATE kernel/${HMLP_ARCH})


# Install libraries.
# ---------------------------
INSTALL(TARGETS hmlp DESTINATION lib)

# Install public headers.
# ---------------------------
INSTALL(DIRECTORY ${hmlp_SOURCE_DIR}/include DESTINATION .)





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
FILE(GLOB TEST_CXX_SRC 
  ${hmlp_SOURCE_DIR}/example/*.cpp)
FOREACH(src ${TEST_CXX_SRC})
  GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
  ADD_EXECUTABLE(${exe} ${src})
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PUBLIC ${INC})
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame)
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE gofmm)
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/base)
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/containers)
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/primitives)
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE kernel/reference)
  TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE kernel/${HMLP_ARCH})
  TARGET_LINK_LIBRARIES(${exe} hmlp ${MPI_CXX_LIBRARIES})
  IF(DEFINED ${BLAS_LIBRARIES})
    TARGET_LINK_LIBRARIES(${exe} hmlp ${BLAS_LIBRARIES})
  ENDIF()
  IF(DEFINED ${LAPACK_LIBRARIES})
    TARGET_LINK_LIBRARIES(${exe} hmlp ${LAPACK_LIBRARIES})
  ENDIF()
  SET_TARGET_PROPERTIES(${exe} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
  SET_TARGET_PROPERTIES(${exe} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
  IF($ENV{HMLP_USE_CUDA} MATCHES "true")
    TARGET_LINK_LIBRARIES(${exe} cudart cublas nvToolsExt)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE "kernel/${HMLP_GPU_ARCH}")
  ENDIF()
  IF($ENV{HMLP_USE_MAGMA} MATCHES "true")
    TARGET_LINK_LIBRARIES(${exe} magma)
  ENDIF()
  INSTALL(TARGETS ${exe} DESTINATION bin)
ENDFOREACH()

# Install GOFMM shell script
# ---------------------------
FILE(GLOB GOFMM_SCRIPTS ${hmlp_SOURCE_DIR}/example/*.sh)
INSTALL(FILES ${GOFMM_SCRIPTS} DESTINATION bin)

## Build paper artifacts
# ---------------------------
SET(ARTIFACT_PATH $ENV{HMLP_ARTIFACT_PATH})
IF(ARTIFACT_PATH)
  SET(ARTIFACT_PATH "${hmlp_SOURCE_DIR}/artifact/${ARTIFACT_PATH}")
  FILE(GLOB ARTIFACT_CXX_SRC ${ARTIFACT_PATH}/*.cpp)
  FILE(GLOB ARTIFACT_SCRIPTS ${ARTIFACT_PATH}/*.sh)
  FILE(GLOB ARTIFACT_DATASETS ${ARTIFACT_PATH}/datasets/*.m)
  FOREACH(src ${ARTIFACT_CXX_SRC} )
    GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
    ADD_EXECUTABLE(${exe} ${src})
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PUBLIC ${INC})
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE gofmm)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/base)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/containers)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/primitives)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE kernel/reference)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE kernel/${HMLP_ARCH})
    TARGET_LINK_LIBRARIES(${exe} hmlp ${MPI_CXX_LIBRARIES})
    IF(DEFINED ${BLAS_LIBRARIES})
      TARGET_LINK_LIBRARIES(${exe} hmlp ${BLAS_LIBRARIES})
    ENDIF()
    IF(DEFINED ${LAPACK_LIBRARIES})
      TARGET_LINK_LIBRARIES(${exe} hmlp ${LAPACK_LIBRARIES})
    ENDIF()
    SET_TARGET_PROPERTIES(${exe} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
    SET_TARGET_PROPERTIES(${exe} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    IF($ENV{HMLP_USE_CUDA} MATCHES "true")
      TARGET_LINK_LIBRARIES(${exe} cudart cublas nvToolsExt)
      TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PUBLIC "${CUDA_TOOLKIT_ROOT_DIR}/include")
      TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE "kernel/${HMLP_GPU_ARCH}")
    ENDIF()
    IF($ENV{HMLP_USE_MAGMA} MATCHES "true")
      TARGET_LINK_LIBRARIES(${exe} magma)
    ENDIF()
    INSTALL(TARGETS ${exe} DESTINATION bin)
  ENDFOREACH()
  INSTALL(FILES ${ARTIFACT_SCRIPTS} DESTINATION bin)
  INSTALL(FILES ${ARTIFACT_DATASETS} DESTINATION bin/datasets)
ENDIF()


# ---[ Build google test
IF(BUILD_MOCK_TESTS OR BUILD_UNIT_TESTS)
  IF(NOT TARGET gtest)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
    INCLUDE_DIRECTORIES("${GOOGLETEST_SOURCE_DIR}/googletest/include")
  ENDIF()
  FILE(GLOB TEST_CXX_SRC 
    ${hmlp_SOURCE_DIR}/test/*.cpp 
    ${hmlp_SOURCE_DIR}/test/*.cu)
  FOREACH(src ${TEST_CXX_SRC})
    GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
    ADD_EXECUTABLE(${exe} ${src})
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PUBLIC ${INC})
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE gofmm)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/base)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/containers)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE frame/primitives)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE kernel/reference)
    TARGET_INCLUDE_DIRECTORIES(${exe} BEFORE PRIVATE kernel/${HMLP_ARCH})
    TARGET_LINK_LIBRARIES(${exe} hmlp ${MPI_CXX_LIBRARIES} gtest)
    IF(DEFINED ${BLAS_LIBRARIES})
      TARGET_LINK_LIBRARIES(${exe} hmlp ${BLAS_LIBRARIES})
    ENDIF()
    IF(DEFINED ${LAPACK_LIBRARIES})
      TARGET_LINK_LIBRARIES(${exe} hmlp ${LAPACK_LIBRARIES})
    ENDIF()
    IF("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
      TARGET_LINK_LIBRARIES(${exe} hmlp ${MPI_CXX_LIBRARIES} gcov)
    ENDIF()
    SET_TARGET_PROPERTIES(${exe} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
    SET_TARGET_PROPERTIES(${exe} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    ADD_TEST(${exe} ${exe})  
    INSTALL(TARGETS ${exe} DESTINATION bin)
  ENDFOREACH()
ENDIF()


ADD_CUSTOM_TARGET(coverage
  COMMAND gcovr -r ${hmlp_SOURCE_DIR} -e ${hmlp_SOURCE_DIR}/deps -e ${hmlp_SOURCE_DIR}/example -e ${hmlp_SOURCE_DIR}/test) 
ADD_CUSTOM_TARGET(coverage-html
  COMMAND gcovr -r ${hmlp_SOURCE_DIR} -e deps -e example -e test --html -o ${hmlp_SOURCE_DIR}/doc/html/coverage.html
  COMMAND gcovr -r ${hmlp_SOURCE_DIR} -e deps -e example -e test --html --html-details -o ${hmlp_SOURCE_DIR}/doc/html/coverage_details.html)



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
message("EXE Linker   =${CMAKE_SHARED_LINKER_FLAGS}")
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
