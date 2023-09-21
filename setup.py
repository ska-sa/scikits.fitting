#!/usr/bin/env python

###############################################################################
# Copyright (c) 2007-2018, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import os

from setuptools import dist, find_packages

# Ensure we have numpy before we start as it is needed before we call setup()
# If not installed system-wide it will be downloaded into the local .eggs dir
dist.Distribution(dict(setup_requires='numpy'))

from numpy.distutils.core import setup  # noqa: E402 (needs pre-setup above)
from numpy.distutils.misc_util import Configuration  # noqa: E402


def configuration(parent_package='', top_path=None):
    """Configuration to deal with scikits namespace."""
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


here = os.path.abspath(os.path.dirname(__file__))
readme = open(os.path.join(here, 'README.rst')).read()
news = open(os.path.join(here, 'NEWS.rst')).read()
long_description = readme + '\n\n' + news

setup(name='scikits.fitting',
      description='Framework for fitting functions to data with SciPy',
      long_description=long_description,
      maintainer='Ludwig Schwardt',
      maintainer_email='ludwig@ska.ac.za',
      url='https://github.com/ska-sa/scikits.fitting',
      license='Modified BSD',
      version='0.7.1',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'Topic :: Scientific/Engineering'],
      configuration=configuration,
      platforms=['OS Independent'],
      python_requires='>=2.7, !=3.0, !=3.1, !=3.2, <4',
      install_requires=['numpy', 'scipy>=0.9', 'future'],
      namespace_packages=['scikits'],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
