"""
Submodule for neural network model and training,
applyied to error estimation problems.
Using tensorflow and keras.
"""
import numpy as np
import tensorflow as tf


class NeuralNetwork:
    """
    Class for neural network model and training.
    """
    def __init__(
            self,
            shape: tuple,
            hidden_layers: int,
            hidden_units: int,
            activation: str = 'relu',
            optimizer: str = 'adam',
            loss: str = 'mean_squared_error') -> None:
        """
        Constructor for NeuralNetwork class.
        """
        self.input_shape = shape[0]
        self.output_shape = shape[1]
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.model = self.create_model(activation, optimizer, loss)

    def create_model(
            self,
            activation: str = 'relu',
            optimizer: str = 'adam',
            loss: str = 'mean_squared_error') -> tf.keras.models.Sequential:
        """
        Create a neural network model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        for _ in range(self.hidden_layers):
            model.add(
                    tf.keras.layers.Dense(
                        self.hidden_units,
                        activation=activation))
        model.add(tf.keras.layers.Dense(self.output_shape))
        model.compile(optimizer=optimizer, loss=loss)
        return model

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

    def predict(
            self,
            x: np.ndarray) -> np.ndarray:
        """
        Predict the output for input x.
        """
        return self.model.predict(x)

    def evaluate(
            self,
            x: np.ndarray,
            y: np.ndarray):
        """
        Evaluate the model on input x and output y.
        """
        return self.model.evaluate(x, y)

    def save(
            self,
            filename: str) -> None:
        """
        Save the model to a file.
        """
        self.model.save(filename)

    def load(
            self,
            filename: str) -> None:
        """
        Load the model from a file.
        """
        self.model = tf.keras.models.load_model(filename)
