"""Top-level package for caloutils."""

__author__ = """mova"""
__email__ = "mova@users.noreply.github.com"
__version__ = '0.0.9'  # fmt: skip

from .calorimeter import calorimeter


def init_calorimeter(caloname: str):
    """The function `init_calorimeter` initializes a calorimeter geometry.
    Currently implemented are dataset 2 and 3 of the CaloChallenge:
    https://github.com/CaloChallenge/homepage

    Parameters
    ----------
    caloname : str
        The parameter `caloname` is a string that represents the name of the calorimeter. It can have two
    possible values: "ccds2" or "ccds3".

    """
    match caloname:
        case "cc_ds2":
            calorimeter.set_layout_calochallange_ds2()
        case "cc_ds3":
            calorimeter.set_layout_calochallange_ds3()
        case _:
            raise NotImplementedError(
                f"No such calorimeter: {caloname}. Options are :'cc_ds2','cc_ds3'"
            )
