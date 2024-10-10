"""
Implementation of an abstract class Estimator, which is used to estimate the
error of a numerical method.
"""
import numpy as np
from typing import Any
from ngs_ee.adap_strat import AdaptivityStrategy


def mark_from_binary(
        estimator: np.ndarray,
        mesh) -> None:
    """
    Mark elements for refinement.
    INPUTS:
    * estimator: Binary array to mark elements for refinement.
    * mesh: Mesh to mark.
    """
    for element in mesh.Elements():
        mesh.SetRefinementFlag(element, estimator[element.nr] > 0)


class Estimator:
    """
    Abstract class for error estimators.
    """
    def __init__(
            self,
            local_error,
            adaptivity_strategy: AdaptivityStrategy,
            description: str = 'Abstract error estimator'):
        """
        Constructor of the abstract class Estimator.
        """
        self.local_error = local_error
        self.adaptivity_strategy = adaptivity_strategy
        self.description = description

    def __call__(self, *args, **kwargs) -> Any:
        """
        Abstract method for executing the error estimation.
        """
        return [self.local_error(arg, **kwargs) for arg in args]

    def mark(
            self,
            mesh,
            *args,
            **kwargs) -> None:
        """
        Mark elements for refinement.
        INPUTS:
        * mesh: Mesh to mark.
        * args: NGsolve functions to compute the error estimator.
        * kwargs: Keyword arguments to pass to the strategy.
        """
        if getattr(self, 'verbose', True):
            print(f'Description:\n{self.description}\n')
        self.__mark(mesh, *args, **kwargs)

    def __mark(
            self,
            mesh,
            *args,
            **kwargs) -> None:
        """
        Abstract method for marking elements for refinement.
        INPUTS:
        * mesh: Mesh to mark.
        * adaptivity_strategy: Adaptivity strategy to use.
        * kwargs: Keyword arguments to pass to the strategy.
        """
        print(f'Applying {self.adaptivity_strategy} adaptivity strategy'
              f' to the mesh {mesh}.')
        if self.adaptivity_strategy is AdaptivityStrategy.UNIFORM:
            self.__uniform(mesh, **kwargs)
        elif self.adaptivity_strategy is AdaptivityStrategy.GREEDY:
            self.__greedy(mesh, *args, **kwargs)
        elif self.adaptivity_strategy is AdaptivityStrategy.AVERAGE:
            self.__average(mesh, *args, **kwargs)
        elif self.adaptivity_strategy is AdaptivityStrategy.FIXED_RATES:
            self.__fixed_rates(mesh, *args, **kwargs)
        elif self.adaptivity_strategy is AdaptivityStrategy.EXTRAPOLATION:
            self.__extrapolation(mesh, *args, **kwargs)
        else:
            raise ValueError('Unknown adaptivity strategy.')

    def __compute_average_estimator(
            self,
            mesh,
            *args,
            **kwargs) -> np.ndarray:
        """
        Compute the average error estimator.
        INPUTS:
        * mesh: Mesh to compute the error estimator.
        * args: NGsolve functions to compute the error estimator.
        """
        estimators = self(*args, **kwargs)
        estimator = np.array(
                [sum((est[element.nr] for est in estimators))
                 / len(list(mesh.Elements()))
                 for element in mesh.Elements()])
        return estimator

    def __compute_dominating_estimator(
            self,
            mesh,
            *args,
            **kwargs) -> np.ndarray:
        """
        Compute the dominating (max) error estimator.
        INPUTS:
        * mesh: Mesh to compute the error estimator.
        * args: NGsolve functions to compute the error estimator.
        """
        estimators = self(*args, **kwargs)
        estimator = np.array(
                [max((est[element.nr] for est in estimators))
                 for element in mesh.Elements()])
        return estimator

    def __uniform(
            self,
            mesh,
            **kwargs) -> None:
        """
        Mark all elements for refinement.
        INPUTS:
        * mesh: Mesh to mark.
        """
        mesh.Refine(**kwargs)

    def __greedy(
            self,
            mesh,
            *args,
            weight: float = 0.9,
            **kwargs) -> None:
        """
        Take the maximum of the error estimators.
        INPUTS:
        * weight: Weight to multiply the maximum error estimator by.
        """
        assert 0.0 < weight < 1.0, 'Weight must be between 0 and 1.'
        estimator = self.__compute_dominating_estimator(mesh, *args, **kwargs)
        print(f'Estimator: {type(estimator)}, weight: {type(weight)}')
        threshold = np.max(estimator) * weight
        for element in mesh.Elements():
            mesh.SetRefinementFlag(element, estimator[element.nr] > threshold)

    def __average(
            self,
            mesh,
            *args,
            weight: float = 1.0,
            shift: float = 0.0,
            **kwargs) -> None:
        """
        Take the average of the error estimators.
        We can implemen an skewed average by using a weight and a shift.
        The threshold is defined by
            threshold = (sum(estimator) / len(estimator) + shift) * weight
        INPUTS:
        * weight: Weight to multiply the average error estimator by.
        * shift: Shift to add to the average error estimator.
        """
        assert 0.0 < weight <= 1.0, 'Weight must be between 0 and 1.'
        assert 0.0 <= shift, 'Shift must be non-negative.'
        estimator = self.__compute_average_estimator(mesh, *args, **kwargs)
        threshold = (np.sum(estimator) / np.len(estimator) + shift) * weight
        for element in mesh.Elements():
            mesh.SetRefinementFlag(element, estimator[element.nr] > threshold)

    def __fixed_rates(
            self,
            mesh,
            *args,
            rate: float = 0.5,
            **kwargs) -> None:
        """
        Use fixed rates to determine the refinement criteria.
        (Cf. Heuveline and Rannacher, 2001)
        The goal is to either increase the refinements by a fixed rate
        or to reduce the error estimators by a fixed rate.
        INPUTS:
        * rate: Fixed rate of elements to refine.
        """
        assert 0.0 < rate < 1.0, 'Rate must be between 0 and 1.'
        estimator = self.__compute_dominating_estimator(mesh, *args, **kwargs)
        threshold = np.sum(estimator)*rate
        total = 0.0
        # Copy the estimator to avoid modifying the original.
        aux_estimator = estimator.copy()
        while total < threshold:
            index_max = np.argmax(aux_estimator)
            max_elem = list(iter(mesh.Elements()))[index_max]
            mesh.SetRefinementFlag(max_elem, True)
            total += aux_estimator[index_max]
            if total >= threshold:
                break
            aux_estimator[index_max] = 0.0

    def __extrapolation(self, mesh, estimator):
        """
        Use extrapolation to determine the refinement criteria.
        (Cf. Verfürth, 1996)
        INPUTS:
        """
        raise NotImplementedError('Extrapolation not implemented yet.')
        # assert self.aux_estimator is not None, 'Aux estimator not set.'
# EOF estimator.py
