"""Top-level package for caloutils."""

__author__ = """mova"""
__email__ = "moritz.scham@desy.de"
__version__ = '0.0.13'  # fmt: skip

from .calorimeter import calorimeter

init_calorimeter = calorimeter.init_calorimeter
from . import distances, processing, utils, variables
