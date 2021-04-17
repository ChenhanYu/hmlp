# Use 10.1-ubuntu16.04 as base
FROM nvidia/cuda:11.2.2-devel-ubuntu18.04

# Developer
MAINTAINER Chenhan D. Yu <chenhany@utexas.edu>

# Set the working directory to /app
WORKDIR /hmlp

# Copy the current directory contents into the container at /app
COPY . /hmlp

# Remove the build directory
RUN rm -rf build

# Install dependencies through apt-get.
RUN apt-get update && apt-get install -y -qq \
    git \
    wget \
    vim \
    build-essential \
    ca-certificates \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libnuma-dev \
    libpmi2-0-dev 

# Install CMAKE-3.14
ENV CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz" 
ENV CMAKE_ROOT="/hmlp/.dependencies/cmake" 
RUN mkdir -p $CMAKE_ROOT 
RUN wget --no-check-certificate --quiet -O - $CMAKE_URL | tar --strip-components=1 -xz -C $CMAKE_ROOT 

# Install UCX + OpenMPI
WORKDIR /openmpi
ENV OPENMPI_VERSION=3.1.4
ENV UCX_URL="https://github.com/openucx/ucx/releases/download/v1.4.0/ucx-1.4.0.tar.gz"

RUN wget -q -O - ${UCX_URL} | tar --strip-components=1 -xzf - \
    && ./configure --prefix=/usr/local --with-cuda=/usr/local/cuda \
    && make -j"$(nproc)" install >/dev/null

RUN wget -q -O - https://www.open-mpi.org/software/ompi/v$(echo "${OPENMPI_VERSION}" | cut -d . -f 1-2)/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar --strip-components=1 -xzf - \
    && ./configure --prefix=/usr/local --disable-getpwuid --with-cuda=/usr/local/cuda \
    && make -j"$(nproc)" install >/dev/null


# Environment variables
ENV PATH="${CMAKE_ROOT}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CC="gcc"
ENV CXX="g++"
ENV CUDACXX="/usr/local/cuda/bin/nvcc"
ENV OPENBLASROOT="/usr/lib/x86_64-linux-gnu"

# Build HMLP (develop build)
WORKDIR /hmlp
