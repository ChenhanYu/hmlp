from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy.distutils.misc_util

# #include directories
inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/include']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/primitives']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/containers']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/gofmm']
print inc_dirs

# hmlp library directory
lib_dirs = ['${CMAKE_BINARY_DIR}/lib']
print lib_dirs





# the c++ extension module
extension_mod_hmlp = Extension( 
        "hmlp", 
        sources = ['${CMAKE_BINARY_DIR}/python/hmlp.pyx'], 
				language="c++",
        include_dirs = inc_dirs,
        libraries = ['hmlp'],
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs,
				extra_compile_args=["-fopenmp", "-O3", "-std=c++11"],
				#extra_compile_args=["-fopenmp", "-O3", "-std=c++11",
				#	"-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
				extra_link_args=[""])



setup(
  author = 'Chenhan D. Yu',
  author_email = 'chenhan@cs.utexas.edu',
  ext_modules = cythonize([extension_mod_hmlp]) )


# the c++ extension module
extension_mod_gofmm = Extension( 
        "gofmm", 
        sources = ['${CMAKE_BINARY_DIR}/python/gofmm.pyx'], 
				language="c++",
        include_dirs = inc_dirs,
        libraries = ['hmlp'],
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs,
				extra_compile_args=["-fopenmp", "-O3", "-std=c++11"],
				#extra_compile_args=["-fopenmp", "-O3", "-std=c++11",
				#	"-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"],
				extra_link_args=[""])


setup(
  author = 'Chenhan D. Yu',
  author_email = 'chenhan@cs.utexas.edu',
  ext_modules = cythonize([extension_mod_gofmm]) )
