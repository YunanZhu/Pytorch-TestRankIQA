import numpy as np


def plcc(x, y):
    """
    Pearson’s linear correlation coefficient.

    :param x: score vector 1, with n raw scores.
    :param y: score vector 2, with n raw scores.
    :return: PLCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise Exception("Please input N (* 1) vector.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The lengths of 2 input vectors are not equal.")

    x = x - np.average(x)
    y = y - np.average(y)
    numerator = np.dot(x, y)
    denominator = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    return numerator / denominator


def srocc(x, y):
    """
    Spearman's rank order correlation coefficient.

    :param x: score vector 1, with n raw scores.
    :param y: score vector 2, with n raw scores.
    :return: SROCC of 2 input vectors (x and y).
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    if len(x.shape) != 1 or len(y.shape) != 1:
        raise Exception("Please input N (* 1) vector.")
    if x.shape[0] != y.shape[0]:
        raise Exception("The lengths of 2 input vectors are not equal.")

    rank_x = x.argsort().argsort()
    rank_y = y.argsort().argsort()
    return plcc(rank_x, rank_y)


if __name__ == "__main__":
    test_a = [2.0, 3.0, 4.0]
    test_b = [1.0, 5.0, 5.3]
    print(plcc(test_a, test_b))
    print(srocc(test_a, test_b))
