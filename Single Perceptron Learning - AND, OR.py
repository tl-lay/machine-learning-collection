"""
Learning perceptrons to represent AND logic and OR logic

Terry Lay
"""
import random


def compute_output(w, x):
    """
    Compute the output for a single input of the perceptron
    :param w: The weights
    :param x: A single input from the dataset
    :return: The output of the sign function based on w and x
    """
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]  # Compute sum of weighted inputs
    if z < 0:  # Apply sign function
        return -1
    else:
        return 1


def perceptron_learning(X, y, learning_rate):
    """
    Trains a single perceptron.
    :param X: The training input
    :param y: The ground truth labels
    :param learning_rate: The step size at each iteration
    :return: The trained weights
    """
    # Choose random starting weights from the integers -1, 0, or 1.
    w = [random.randint(-1, 1), random.randint(-1, 1), random.randint(-1, 1)]
    # Compute predicted labels based on the starting weights
    y_hat = []
    for i in range(len(y)):
        y_hat.append(compute_output(w, X[i]))
    # Loop until the predicted labels are equal to ground truth
    while y_hat != y:
        # Choose a random input/output pair
        choice = random.randint(0, len(y) - 1)
        x = X[choice]
        y_i = y[choice]
        y_hat_i = compute_output(w, x)
        # print(x, w)
        if y_hat_i != y_i:
            # Update weights if input/output do not match ground truth
            for i in range(len(w)):
                w[i] += y_i * learning_rate * x[i]
        # Recalculate the predicted labels based on the new weights
        for i in range(len(y)):
            y_hat[i] = compute_output(w, X[i])
    return w


# Define matrix X and learning rate
x0 = 1
X = [[x0, 1, 1], [x0, 1, -1], [x0, -1, 1], [x0, -1, -1]]
learning_rate = 1

# For the AND logic
y_AND = [1, -1, -1, -1]
w_AND = perceptron_learning(X, y_AND, learning_rate)
y_AND_hat = []
for i in range(len(X)):
    y_AND_hat.append(compute_output(w_AND, X[i]))
print("Weights for AND: {} \n"
      "Y_hat based on weights for AND: {} \n"
      "Does predicted equal ground truth?: {}".format(w_AND, y_AND_hat, y_AND_hat == y_AND))

print('\n')

# For the OR logic
y_OR = [1, 1, 1, -1]
w_OR = perceptron_learning(X, y_OR, learning_rate)
y_OR_hat = []
for i in range(len(X)):
    y_OR_hat.append(compute_output(w_OR, X[i]))
print("Weights for OR: {} \n"
      "Y_hat based on weights for OR: {}\n"
      "Does Predicted equal ground truth?: {}".format(w_OR, y_OR_hat, y_OR_hat == y_OR))
