import math
import random


def uniform(size=1):
    """
    Generate U ~ Uniform(0, 1) samples.
    Returns a list of `size` samples.
    """
    return [random.random() for _ in range(size)]


def exponential(mean=1.0, size=1):
    """
    Generate X ~ Exponential with E[X] = mean.
    Method: Inverse CDF — X = -mean * ln(U), where U ~ Uniform(0,1).
    """
    return [-mean * math.log(random.random()) for _ in range(size)]


def poisson(mean=1.0, size=1):
    """
    Generate Z ~ Poisson with E[Z] = mean (lambda = mean).
    Method: Knuth algorithm — multiply uniforms until product < e^(-lambda).
    The count of multiplications minus 1 is the Poisson variate.
    """
    L = math.exp(-mean)
    samples = []
    for _ in range(size):
        k = 0
        p = 1.0
        while p > L:
            p *= random.random()
            k += 1
        samples.append(k - 1)
    return samples
