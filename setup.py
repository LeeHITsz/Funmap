# setup.py

import os
from setuptools import setup, find_packages

__version__ = '1.0.0'

setup(
    name='funmap',
    version=__version__,
    author='Yuekai Li',
    author_email='leeykhitsz@gmail.com',
    url='https://github.com/LeeHITsz/Funmap',
    description='A method for fine-mapping with functional annotations.',
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=[
        'matplotlib>=3.8.2',
        'numpy>=1.26.2',
        'pandas>=2.1.3',
        'scipy>=1.11.4'
    ],
    license="MIT license",
    zip_safe=False
)
