"""
Feed-forward neural network model for an error estimator,
based on TensorFlow and designed for NGSolve.
Cf. Gillete et at.
"""
import tensorflow as tf
import numpy as np
import ngsolve as ng
from ngs_ee.estimators import Estimators, mark_from_binary
from ngs_ee.adap_strat import AdaptivityStrategy
from ngs_ee.nn import NeuralNetwork


class NNEstimator(Estimators, NeuralNetwork):
    """
    Feed-forward neural network model for an error estimator,
    based on TensorFlow and designed for NGSolve.
    """
    def __init__(
            self,
            local_error,
            adaptivity_strategy,
            description: str = 'Neural network error estimator',
            shape: tuple = (3, 2),
            hidden_layers: int = 3,
            hidden_units: int = 10,
            activation: str = 'relu',
            optimizer: str = 'adam',
            loss: str = 'mean_squared_error') -> None:
        """
        Constructor of the class NNEstimator.
        """
        Estimators.__init__(
                self,
                local_error,
                adaptivity_strategy,
                description)
        NeuralNetwork.__init__(
                self,
                shape,
                hidden_layers,
                hidden_units,
                activation=activation,
                optimizer=optimizer,
                loss=loss)

    def __call__(self, fes: ng.FESpace, *args, **kwargs) -> np.ndarray:
        """
        Execute the error estimation.
        Returns the sample parameter and the estimators (from the base class).
        """
        estimators = Estimators.__call__(self, *args, **kwargs)
        normalized_estimator, emp_expectation, emp_std_deviation = \
            self.__compute_input(fes, estimators, **kwargs)
        mu, ln_sigma = self.model.predict(
                np.array([normalized_estimator,
                          emp_expectation,
                          emp_std_deviation]))
        sample = self.__compute_output(mu, ln_sigma, **kwargs)
        return sample, estimators

    def __compute_input(
            self,
            fes: ng.FESpace,
            *args,
            **kwargs) -> tuple(float, float, float):
        """
        Compute the empirical expectation of the error estimator.
        It consider the dominating estimator as the input.
        This will be fed into the neural network.
        """
        estimator = Estimators.__compute_dominating_estimator(*args, **kwargs)
        zeta = np.zeros(fes.mesh.ne)
        for element in fes.mesh.Elements():
            zeta[element.nr] = -np.log(fes.mesh.ne ** .5 *
                                       estimator[element.nr]) \
                                       / np.log(fes.ndofs)
        normalized_estimator = np.sum(estimator) / np.max(estimator)
        emp_expectation = np.mean(zeta)
        emp_std_deviation = np.std(zeta)
        return normalized_estimator, emp_expectation, emp_std_deviation

    def __compute_output(
            self,
            mu: float,
            ln_sigma: float) -> np.ndarray:
        """
        From the neural network, compute the Gaussian sample.
        """
        rand_proj = np.max([0, np.min([1, np.random.normal()])])
        sigma = np.exp(ln_sigma)
        sample = 1 / np.sqrt(2 * np.pi * sigma ** 2) * \
            np.exp(- (mu - rand_proj) ** 2 / (2 * sigma ** 2))
        return sample

    def mark(
            self,
            mesh,
            *args,
            **kwargs) -> None:
        """
        Mark elements for refinement.
        TODO: Simplify this method.
        """
        sample, _ = self(*args, **kwargs)
        if self.adaptivity_strategy is AdaptivityStrategy.UNIFORM:
            raise ValueError('Uniform adaptivity strategy'
                             ' is not supported for neural network estimator.')
        if self.adaptivity_strategy is AdaptivityStrategy.GREEDY:
            self.__greedy(mesh, *args, weight=sample, **kwargs)
        elif self.adaptivity_strategy is AdaptivityStrategy.AVERAGE:
            raise ValueError('Average adaptivity strategy'
                             ' is not supported for neural network estimator.')
        elif self.adaptivity_strategy is AdaptivityStrategy.FIXED_RATES:
            raise ValueError('Fixed rates adaptivity strategy'
                             ' is not supported for neural network estimator.')
        elif self.adaptivity_strategy is AdaptivityStrategy.EXTRAPOLATION:
            raise ValueError('Extrapolation adaptivity strategy'
                             ' is not supported for neural network estimator.')
        else:
            raise ValueError('Unknown adaptivity strategy.')

    def train(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int) -> None:
        """
        Train the neural network model.
        """
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
