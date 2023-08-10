#!/usr/bin/env python

from setuptools import setup, find_packages


# This setup is suitable for "python setup.py develop".

setup(name='moleculelocator',
      version='0.0.1',
      description='Script containing classes and helper functions for molecule location',
      author='Koen Lauwaet',
      author_email='koen.lauwaet@imdea.org',
      url='https://github.com/KoenImdea/',
      packages=find_packages(),
      install_requires=["numpy", "mahotas", "scipy", "scikit-image", "spiepy", "sklearn"],
)
