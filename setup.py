from setuptools import setup, find_packages

setup(name='sharpesst', version='0.0.0',
    packages=find_packages(include=['sharpesst', 'sharpesst.*']),
    install_requires = ['numpy>=1.14.5',
						'numba>=0.51.1',
						'matplotlib>=3.1.1',
						'scipy>=1.5',
						'astropy>=4.1.1']
)
