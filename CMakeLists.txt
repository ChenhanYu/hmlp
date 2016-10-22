# Check cmake version.
cmake_minimum_required (VERSION 2.8)


# Project specification.
project (HMLP)
set (HMLP_VERSION_MAJOR 1)
set (HMLP_VERSION_MINOR 1)
set (HMLP_VERSION_PATCH 0)
set (HMLP_VERSION ${HMLP_VERSION_MAJOR}.${HMLP_VERSION_MINOR}.${HMLP_VERSION_PATCH})


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


# Compiler Options (HMLP can use Intel or GNU compilers.)
if ($ENV{HMLP_USE_INTEL} MATCHES "true")
  set (CMAKE_C_COMPILER   icc )
  set (CMAKE_CXX_COMPILER icpc)
  set (HMLP_CFLAGS          "-O3 -qopenmp -mavx -Werror -Wall -pedantic")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -qopenmp -lm")
else ($ENV{HMLP_USE_INTEL} MATCHES "true")
  set (CMAKE_C_COMPILER   gcc)
  set (CMAKE_CXX_COMPILER g++)
  set (HMLP_CFLAGS          "-O3 -fopenmp -mavx")
  set (CMAKE_EXE_LINKER_FLAGS "-lpthread -fopenmp -lm")
endif ($ENV{HMLP_USE_INTEL} MATCHES "true")


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
  set (HMLP_CFLAGS          "${HMLP_CFLAGS} -DUSE_BLAS")
endif ($ENV{HMLP_USE_BLAS} MATCHES "true")
set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS} ${HMLP_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${HMLP_CFLAGS}")


# Linking infos.
if (($ENV{HMLP_USE_BLAS} MATCHES "true") OR ($ENV{HMLP_USE_VML} MATCHES "true"))
  if ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mkl=parallel")
  else ($ENV{HMLP_USE_INTEL} MATCHES "true")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lblas")
  endif ($ENV{HMLP_USE_INTEL} MATCHES "true")
endif (($ENV{HMLP_USE_BLAS} MATCHES "true") OR ($ENV{HMLP_USE_VML} MATCHES "true"))


# Software dependencies.
set (MKL_DIR $ENV{HMLP_MKL_DIR})


# Headers.
include_directories ("include" "frame" "${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}" "${MKL_DIR}/include")
file (GLOB HMLP_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${HMLP_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
file (GLOB FRAME_CC_SRC  ${CMAKE_SOURCE_DIR}/frame/*.c)
file (GLOB FRAME_CXX_SRC ${CMAKE_SOURCE_DIR}/frame/*.cpp)
file (GLOB KERNEL_SRC ${CMAKE_SOURCE_DIR}/kernel/${HMLP_ARCH}/*.cpp)


# Build the static library.
add_library (hmlp ${FRAME_CC_SRC} ${FRAME_CXX_SRC} ${KERNEL_SRC})


# Build the executable files.
add_executable (test_hmlp.x ${CMAKE_SOURCE_DIR}/test/test_hmlp.cpp)
target_link_libraries(test_hmlp.x hmlp)
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
message ("MKLDIR       =${MKL_DIR}")
message ("===================================================")
