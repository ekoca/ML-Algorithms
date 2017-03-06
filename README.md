# ML-Algorithms

# INSTRUCTIONS

There are three parts to this assignment. Please read all sections of the instructions carefully. In particular, note that you have a total of 5 submission attempts for each question. 

- Perceptron Learning Algorithm
- Linear Regression with gradient descent
- Classification

## Perceptron Learning Algorithm

Perceptron learning algorithm is one of the oldest and simplest Linear classification method where it belongs to Neural Networks class of algorithms. It works perfectly if data is linearly separable. If not, it will not converge. The idea is to start with a random hyperplane and adjust it using a training data using the iterative method.

The "perceptron.py" have the implementation of the perceptron learning algorithm ("PLA") for a linearly separable dataset. I am using the input.csv, containing a series of data points. Each point is a comma-separated ordered triple, representing `feature_1`, `feature_2`, and the `label` for the point. You can think of the values of the features as the x- and y-coordinates of each point. The label takes on a value of positive or negative one. You can think of the label as separating the points into two categories.

The project uses Python3 and should be executed like so:
```
$ python3 Perceptron.py input.csv output.csv
```
This should generate an output file called output.csv. With each iteration of the PLA, the program will print a new line to the output file, containing a comma-separated list of the weights `w_1`, `w_2`, and `b` in that order. Upon convergence, the program will stop, and the final values of `w_1`, `w_2`, and `b` will be printed to the output file. This defines the decision boundary that the PLA has computed for the given dataset.