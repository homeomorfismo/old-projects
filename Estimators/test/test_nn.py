"""
Basic tests for the neural network.
"""
import numpy as np
from ngs_ee.nn import NeuralNetwork

INPUT_SHAPE: int = 4
OUTPUT_SHAPE: int = 2
HIDDEN_LAYER: int = 2
HIDDEN_UNITS: int = 3
ACTIVATION: str = "relu"
OPTIMIZER: str = "adam"
LOSS: str = "mean_squared_error"

EPOCHS: int = 10
BATCH_SIZE: int = 32


def test_nn():
    """
    Test the neural network.
    """
    nn = NeuralNetwork(
            INPUT_SHAPE,
            OUTPUT_SHAPE,
            HIDDEN_LAYER,
            HIDDEN_UNITS,
            activation=ACTIVATION,
            optimizer=OPTIMIZER,
            loss=LOSS)
    # TODO: Add more assertions
    assert nn.model is not None

    x_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    y_train = np.array([[1, 0], [0, 1]])
    nn.train(x_train, y_train, EPOCHS, BATCH_SIZE)
    output = nn.predict(x_train)
    assert output is not None
    assert output.shape[0] == 2
    assert output.shape[1] == 2
    print(output)
    print("Test passed")

if __name__ == "__main__":
    test_nn()
