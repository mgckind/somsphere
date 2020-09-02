from numpy.distutils.core import setup

exec(open('version.py').read())
setup(
    name='somsphere',
    version=__version__,
    author='Matias Carrasco Kind',
    author_email='mcarras2@illinois.edu',
    packages=[],
    py_modules=['somsphere'],
    license='License.txt',
    description='somsphere : Self Organizing Maps in spherical coordinates and other topologies',
    long_description=open('README.md').read(),
    url='https://github.com/mgckind/somsphere',
    install_requires=['numpy', 'matplotlib', 'scipy'],
)
