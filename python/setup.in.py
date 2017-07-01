from distutils.core import setup, Extension
import numpy.distutils.misc_util


#
inc_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/include']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/primitives']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/containers']
inc_dirs = inc_dirs + ['${CMAKE_SOURCE_DIR}/frame/gofmm']
print inc_dirs

#
lib_dirs = ['${CMAKE_BINARY_DIR}/lib']
print lib_dirs

# the c++ extension module
extension_mod1 = Extension( 
        "hmlp", 
        sources = ['${PROJECT_SOURCE_DIR}/python/hmlpmodule.cpp'], 
        include_dirs = inc_dirs,
        libraries = ['hmlp'],
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs )

print '${CMAKE_SOURCE_DIR}/include'
print '${CMAKE_SOURCE_DIR}/frame'

#
setup(
    name = "hmlp",
    author = 'Chenhan D. Yu',
    author_email = 'chenhan@cs.utexas.edu',
    ext_modules  = [extension_mod1] )




# the c++ extension module
extension_mod2 = Extension( 
        "gofmm", 
        sources = ['${PROJECT_SOURCE_DIR}/python/gofmmmodule.cpp'],
        include_dirs = inc_dirs,
        libraries = ['hmlp'],
        library_dirs = lib_dirs,
        runtime_library_dirs = lib_dirs )

#
setup(
    name = "gofmm",
    author = 'Chenhan D. Yu',
    author_email = 'chenhan@cs.utexas.edu',
    ext_modules  = [extension_mod2] )
