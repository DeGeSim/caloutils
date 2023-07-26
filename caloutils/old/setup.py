#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="mova",
    author_email='mova@users.noreply.github.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Metrics and tools for evaluation generative models for calorimeter shower based on pytorch_geometric.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='caloutils',
    name='caloutils',
    packages=find_packages(include=['caloutils', 'caloutils.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/mova/caloutils',
    version='0.1.0',
    zip_safe=False,
)
