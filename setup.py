#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_namespace_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('docs/HISTORY.md') as history_file:
#     history = history_file.read()

requirements = [
        'pandas>=1.0.3',
        'numba>=0.46.0',
        'numpy>=1.17.4',
        'scipy>=1.3.2',
        'pynwb',
        'tabulate',
        'h5py',
        'tifffile',
        'zarr',
        'rich',
        'pynapple',
        'neuroconv',
        'dandi'
        ]

setup(
    author="John Stout",
    author_email='john.j.stout.jr@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 0.1',
        'Intended Audience :: Science/Research',
        'License :: NA',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="A python toolbox to support data analysis pipeline in the DECODE lab at Nemours Childrens Hospital, led by A.Hernan and R.Scott",
    install_requires=requirements,
    license="GNU General Public License v3",
    # long_description='pynapple is a Python library for analysing neurophysiological data. It allows to handle time series and epochs but also to use generic functions for neuroscience such as tuning curves and cross-correlogram of spikes. It is heavily based on neuroseries.' 
    # + '\n\n' + history,
    long_description=readme,
    include_package_data=True,
    keywords='neuroscience',
    name='decode_lab_code',    
    url='https://github.com/JohnStout/decode_lab_code',
    version='v0.1',
    zip_safe=False,
    long_description_content_type='text/markdown',
    download_url='https://github.com/JohnStout/decode_lab_code'
)
