#! /usr/bin/env python

descr   = """Fitting SciKit.

A framework for fitting functions to data with SciPy which unifies the various
available interpolation methods and provides a common interface to them.
"""

DISTNAME            = 'scikits.fitting'
DESCRIPTION         = 'Framework for fitting functions to data with SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Ludwig Schwardt'
MAINTAINER_EMAIL    = 'ludwig@ska.ac.za'
URL                 = 'https://github.com/ludwigschwardt/scikits.fitting'
LICENSE             = 'Modified BSD'
DOWNLOAD_URL        = 'https://github.com/ludwigschwardt/scikits.fitting'
VERSION             = '0.5'

import os
import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
                           namespace_packages=['scikits'])

    config.set_options(
            ignore_setup_xxx_py=True,
            assume_default_configuration=True,
            delegate_options_to_subpackages=True,
            quiet=True)

    config.add_subpackage('scikits')
    config.add_subpackage(DISTNAME)
    config.add_data_files('scikits/__init__.py')
    config.add_data_dir('scikits/fitting/tests')    

    return config

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,

        classifiers=['Development Status :: 4 - Beta',
                     'Environment :: Console',
                     'Intended Audience :: Developers',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering'],

        configuration=configuration,
        install_requires=[],
        namespace_packages=['scikits'],
        packages=setuptools.find_packages(),
        include_package_data=True,
        zip_safe=False, # the package can run out of an .egg file
        #test_suite="tester", # for python setup.py test
        )
