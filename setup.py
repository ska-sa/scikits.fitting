#!/usr/bin/env python

import os

from setuptools import dist, setup, find_packages

# Ensure we have numpy before we start as it is needed before we call setup()
# If not installed system-wide it will be downloaded into the local .eggs dir
dist.Distribution(dict(setup_requires='numpy'))

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    config = Configuration(None, parent_package, top_path,
                           namespace_packages=['scikits'])

    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('scikits')
    config.add_subpackage('scikits.fitting')
    config.add_data_files('scikits/__init__.py')
    config.add_data_dir('scikits/fitting/tests')

    return config


if __name__ == "__main__":

    with open('README.rst') as readme:
        long_description = readme.read()

    setup(name='scikits.fitting',
          description='Framework for fitting functions to data with SciPy',
          long_description=long_description,
          maintainer='Ludwig Schwardt',
          maintainer_email='ludwig@ska.ac.za',
          url='https://github.com/ska-sa/scikits.fitting',
          license='Modified BSD',
          version='0.5.1',
          classifiers=['Development Status :: 4 - Beta',
                       'Environment :: Console',
                       'Intended Audience :: Developers',
                       'Intended Audience :: Science/Research',
                       'License :: OSI Approved :: BSD License',
                       'Topic :: Scientific/Engineering'],
          configuration=configuration,
          install_requires=['numpy', 'scipy'],
          namespace_packages=['scikits'],
          packages=find_packages(),
          include_package_data=True,
          zip_safe=False)
