=====================
Welcome to caloutils!
=====================

.. image:: https://img.shields.io/pypi/v/caloutils.svg
        :target: https://pypi.python.org/pypi/caloutils

.. .. image:: https://img.shields.io/travis/DeGeSim/caloutils.svg
..         :target: https://travis-ci.com/DeGeSim/caloutils

.. image:: https://readthedocs.org/projects/caloutils/badge/?version=latest
        :target: https://caloutils.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

Metrics and tools for evaluation of generative models for calorimeter showers based on pytorch_geometric.

* Free software: MIT license
* Documentation: https://caloutils.readthedocs.io.
* Github: https://github.com/DeGeSim/caloutils
* PyPi: https://pypi.org/project/caloutils

Summary
=======

``caloutils`` is a Python package built to simplify and streamline the handling, processing, and analysis of 4D point cloud data derived from calorimeter showers in high-energy physics experiments. The package includes a set of sophisticated tools to perform voxelization, energy response calculations, geometric feature extraction, and more. ``caloutils`` aims to simplify the complex analysis pipeline for calorimeter shower data, enabling researchers to efficiently extract meaningful insights. As this tool is based on Point Clouds, the provided metrics should apply to any calorimeter.


Description
===========

4D Point Clouds
---------------


The 4D point cloud data handled by ``caloutils`` consists of three spatial coordinates and a fourth dimension representing the energy deposited at each point in the calorimeter. This multidimensional dataset captures a comprehensive view of particle showers, serving as a valuable resource in experimental physics.

Key Features
------------

``caloutils`` offers a comprehensive suite of functions and methods to analyze these 4D point clouds:

- **Voxelization**: The package provides functionalities to convert raw, continuous point clouds into a to a voxel representation. This regular structure can simplifie subsequent analysis or machine learning tasks.
- **Energy Response Calculation**: Calculate the detector response for a calorimeter shower by summing the hit energies and normalizing by the incoming energy of the particle.
- **Geometric Feature Extraction**: The package offers tools to calculate geometric features such as the first principal component, spherical ratios, and more.
- **Data Transformation**: ``caloutils`` can transform data from cylindrical to Cartesian coordinates, calculate pseudorapidity and azimuthal angle, and efficiently handle batch data operations.

.. With the aforementioned functionalities, ``caloutils`` is an indispensable tool for researchers working with calorimeter shower data.

.. Conclusion
.. ==========

.. Whether you're a particle physicist analyzing complex calorimeter data, a data scientist developing particle detection algorithms, or a computational physicist grappling with high-dimensional data, ``caloutils`` can simplify your workflow and elevate your data analysis capabilities. We encourage you to explore the potential of ``caloutils`` in your research.

Installation
============

You can easily install ``caloutils`` via pip:
.. code-block:: console

   $ pip install caloutils

Usage
=====

First, the used calorimeter geometry needs to be selected:
.. code-block:: python

   import caloutils
   caloutils.init_calorimeter("cc_ds2")

For now only dataset 2 and 3 of the `Calochallenge<https://github.com/CaloChallenge/homepage>`  are implemented

1. Convert Voxelized Data to Point Cloud
----------------------------------------

.. code-block:: python

   import caloutils
   # Convert the point cloud data into a voxel representation.
   batch = caloutils.processing.voxel_to_pc(shower, energies)

``batch`` is an instance of a PyTorch Geometric ``Batch`` object, storing the point cloud data

2. Data Transformation
----------------------

Transform the cylindrical coordinates to Cartesian coordinates and add pseudorapidity and azimuthal angle:

.. code-block:: python

   batch_transformed = caloutils.batch_to_Exyz(batch)

These examples are meant to be illustrative and provide a quick understanding of the package usage. For a more comprehensive understanding of each function's intricacies, users are recommended to refer to the full function documentation in the package.


3. Calculate High Level Variables
---------------------------------
.. code-block:: python

   # Calculate the energy response of a batch of showers.
   energy_response = caloutils.variables.energy_response(batch)
   # Calculate the principal component of a batch of showers.
   first_principal_component = caloutils.variables.fpc_from_batch(batch)
   # Or, all at once, stored as attributes of the batch:
   batch=caloutils.variables.calc_vars(batch)
   print(batch.cyratio.mean())
