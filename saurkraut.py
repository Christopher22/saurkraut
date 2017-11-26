# coding: utf-8

import random
from pathlib import Path

from keras.models import load_model, Sequential
from keras.layers import Dense, GRU

import numpy as np


class DinosaurGenerator:
    def __init__(self, filename=None):
        if filename != None and Path(filename).is_file():
            self._model = load_model(filename)
        else:
            self._model = None

    def is_trained(self):
        return self._model != None

    def save(self, filename):
        if self._model != None:
            self._model.save(filename)
        else:
            raise ValueError('Generator is not trained!')

    def train(self, dinosaur_names, epochs=40):
        x, y = DinosaurGenerator._encode_strings(
            DinosaurGenerator._generate_windows(dinosaur_names, 4))
        x, y = (DinosaurGenerator._encode_categories(x),
                DinosaurGenerator._encode_categories(y))

        self._model = Sequential()
        self._model.add(GRU(x.shape[2], input_shape=(x.shape[1], x.shape[2])))
        self._model.add(Dense(x.shape[2], activation='relu'))

        self._model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

        self._model.fit(x, y, epochs=epochs,
                        batch_size=32,
                        verbose=1)

    def generate(self, start, deterministic=False):
        if self._model is None:
            raise ValueError('Generator is not trained!')

        name = [ord(x) - 96 if x != ' ' else 0 for x in start.rjust(3)]

        while name[-1] != 0:
            current_fragment = np.array([name[-3:]])
            probabilities = self._model.predict(
                DinosaurGenerator._encode_categories(current_fragment, 27))[0]

            if not deterministic:
                most_likely = np.argsort(probabilities)[-3:][::-1]
                name.append(random.choice(
                    17 * [most_likely[0]] + 2 * [most_likely[1]] + 1 * [most_likely[2]]))
            else:
                name.append(np.argmax(probabilities))

        return ''.join((chr(c + 96) for c in name if c != 0))

    @staticmethod
    def _encode_categories(matrix, max_value=None):
        max_value = max_value if max_value else matrix.max() + 1
        if matrix.ndim == 1:
            result = np.zeros((matrix.size, max_value), dtype=np.uint8)
            result[np.arange(matrix.size), matrix] = 1
            return result
        elif matrix.ndim == 2:
            return (np.arange(max_value) == matrix[..., None]).astype(np.uint8)
        else:
            raise ValueError("Dimension unsupported")

    @staticmethod
    def _generate_windows(dataset, max_window_size):
        result = []
        for current_win_size in range(2, max_window_size + 1):
            for data in dataset:
                result.extend(
                    (
                        data[i:i + current_win_size].rjust(max_window_size)
                        for i in range(len(data) - (current_win_size - 1))
                    )
                )
        return result

    @staticmethod
    def _encode_strings(dataset):
        x = np.zeros((len(dataset), len(dataset[0]) - 1), dtype=np.uint8)
        y = np.zeros((len(dataset)), dtype=np.uint8)
        for i, data in enumerate(dataset):
            x[i] = [ord(x) - 96 if x != ' ' else 0 for x in data[:-1]]
            y[i] = ord(data[-1]) - 96 if data[-1] != ' ' else 0
        return (x, y)


def read_file(filename):
    with open(filename) as f:
        return [x.strip().lower() + ' ' for x in f.readlines()]


generator = DinosaurGenerator('model.h5')
if not generator.is_trained():
    generator.train(read_file('dinosaur.txt'))
    generator.save('model.h5')

generator.generate('a')
