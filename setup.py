import sys
files_f90 = ['som.f90', ]
from numpy.distutils.core import setup, Extension

extra_link_args = []
libraries = []
library_dirs = []
exec(open('version.py').read())
setup(
    name = 'somsphere',
    version = __version__,
    author = 'Matias Carrasco Kind',
    author_email = 'mcarras2@illinois.edu',
    ext_modules = [Extension('somF', files_f90, ), ],
    packages = [],
    py_modules = ['somsphere'],
    license = 'License.txt',
    description = 'somsphere : Self Organizing Maps in spherical coordinates and other topologies',
    long_description = open('README.md').read(),
    url='https://github.com/mgckind/somsphere',
    install_requires=['numpy', 'matplotlib', 'scipy'],
)
