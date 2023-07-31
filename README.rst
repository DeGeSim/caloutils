=========
caloutils
=========


.. image:: https://img.shields.io/pypi/v/caloutils.svg
        :target: https://pypi.python.org/pypi/caloutils

.. image:: https://img.shields.io/travis/DeGeSim/caloutils.svg
        :target: https://travis-ci.com/DeGeSim/caloutils

.. image:: https://readthedocs.org/projects/caloutils/badge/?version=latest
        :target: https://caloutils.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Metrics and tools for evaluation generative models for calorimeter shower based on pytorch_geometric.

* Free software: MIT license
* Documentation: https://caloutils.readthedocs.io.



Summary
=======

``caloutils`` is a Python package built to simplify and streamline the handling, processing, and analysis of 4D point cloud data derived from calorimeter showers in high-energy physics experiments. The package includes a set of sophisticated tools to perform voxelization, energy response calculations, geometric feature extraction, and more. ``caloutils`` aims to simplify the complex analysis pipeline for calorimeter shower data, enabling researchers to efficiently extract meaningful insights.

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

With the aforementioned functionalities, ``caloutils`` is an indispensable tool for researchers working with calorimeter shower data.

Conclusion
==========

Whether you're a particle physicist analyzing complex calorimeter data, a data scientist developing particle detection algorithms, or a computational physicist grappling with high-dimensional data, ``caloutils`` can simplify your workflow and elevate your data analysis capabilities. We encourage you to explore the potential of ``caloutils`` in your research.

Installation
============

You can easily install ``caloutils`` via pip::

   pip install caloutils

Usage
=====

First it is necessary to setup the calorimeter geometry. This can be done::

   from caloutils.calorimeter import Calorimeter
   Calorimeter.set_layout_calochallange_ds2()

For now only dataset 2 and 3 of the Calochallenge are implemented

1. Voxelization of Point Cloud Data
-----------------------------------

Assume ``batch`` is an instance of a PyTorch Geometric ``Batch`` object, storing the point cloud data::

   import caloutils
   # Convert the point cloud data into a voxel representation.
   vox_data = caloutils.voxelize(batch)

2. Calculating Energy Response
-------------------------------

Assume ``batch`` is an instance of a ``Batch`` object, storing the point cloud data::

   import caloutils
   # Calculate the energy response of a shower.
   energy_response = caloutils.energy_response(batch)

3. Extracting Geometric Features
--------------------------------

Assume ``batch`` is an instance of a ``Batch`` object, storing the point cloud data::

   import caloutils
   # Extract the first principal component of a point cloud.
   first_principal_component = caloutils.fpc_from_batch(batch)

4. Data Transformation
----------------------

Assume ``batch`` is an instance of a ``Batch`` object, storing the point cloud data::

   import caloutils
   # Transform the cylindrical coordinates to Cartesian coordinates and add pseudorapidity and azimuthal angle.
   batch_transformed = caloutils.batch_to_Exyz(batch)

These examples are meant to be illustrative and provide a quick understanding of the package usage. For a more comprehensive understanding of each function's intricacies, users are recommended to refer to the full function documentation in the package.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
