"""
Enumerators for the marking strategies.
"""
from enum import Enum


class AdaptivityStrategy(Enum):
    """
    Enumerator for the adaptivity strategies.
    """
    UNIFORM = 0
    GREEDY = 1
    AVERAGE = 2
    FIXED_RATES = 3
    EXTRAPOLATION = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        """
        Get the enumerator from the string.
        Used with argparse.
        INPUTS:
        * s: String to convert to enumerator.
        """
        try:
            return AdaptivityStrategy[s]
        except KeyError as exc:
            raise ValueError() from exc
