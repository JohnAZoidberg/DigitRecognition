# DigitRecognition
Very simple feed-forward neural network

It is trained and tested using the MNIST dataset of handwritten digits.

## Building

Prerequisites: Gradle and JDK

``
gradle build
``

## Running
``
gradle run --args draw
``
The possible arguments are `draw`, `eval` and `train`.

- `draw` creates a windows with a canvas for drawing numbers and having them classified
- `train` trains the network with the training data
- `eval` evaluates the performance of the network based on test data

## License
©2015 - 2018 Daniel Schäfer, BSD 3-Clause
