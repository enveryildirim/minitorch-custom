"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable


# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Multiplication function: $f(x, y) = x * y$.
    This will be a fundamental building block for tensor multiplications later.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x * y


def id(x: float) -> float:
    """
    Identity function: $f(x) = x$.
    Returns the input as is. Its derivative is always 1.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x


def add(x: float, y: float) -> float:
    """
    Addition function: $f(x, y) = x + y$
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x + y


def neg(x: float) -> float:
    """
    Negation function: $f(x) = -x$
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return -x


def lt(x: float, y: float) -> float:
    """
    Less than check: 1.0 if $x < y$, else 0.0.
    We use 1.0 and 0.0 to represent logical operations with continuous values.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Equality function: $f(x) = 1.0$ if $x == y$ else $0.0$
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    Maximum function: $f(x) = x$ if $x > y$ else $y$
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Check if two values are close: $f(x) = |x - y| < 1e-2$
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""
    Sigmoid activation function: $f(x) = \frac{1}{1 + e^{-x}}$.

    This function squashes the input between 0 and 1.
    For numerical stability, we use two different formulas based on the sign of x
    to avoid overflow errors with very large positive or negative values.

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    ReLU (Rectified Linear Unit): x if x > 0, else 0.
    This is the most commonly used activation function in neural networks.
    It "kills" negative values while leaving positive ones unchanged.

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    r"""
    Natural logarithm: $\ln(x)$.
    Since log is undefined at x=0, we add EPS for safety.
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """
    Exponential function: $f(x) = e^{x}$
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"""
    Backpropagation derivative for the log function.

    According to the Chain Rule: $dL/dx = dL/df * df/dx$.
    Here $f = \ln(x)$ and $df/dx = 1/x$.
    'd' is the gradient $dL/df$ coming from the upper layer.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return d / (x + EPS)


def inv(x: float) -> float:
    """
    Inverse function: $f(x) = 1/x$
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """
    Gradient of the inverse function: $d \times f'(x) = -d / x^2$.
    The derivative of $1/x$ is $-1/x^2$, which we multiply by the upstream gradient 'd'.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """
    ReLU gradient: $d$ if $x > 0$, else 0.
    Passes the gradient as is for positive values and blocks it for negative ones.
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-Order Function: Map

    Takes a 'function' (fn) and returns a new function.
    The returned function applies 'fn' to each element in a given list.
    This is an excellent example of the concept of abstraction in programming.

    Args:
        fn: A function that takes one value and returns one value.

    Returns:
         A function that takes a list and returns a new list with fn applied to each element.
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return lambda ls: (fn(x) for x in ls)


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate all elements in a list using map and neg.
    Ensures 'map' applies the 'neg' operation to every element in the list.
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-Order Function: ZipWith (or Map2)

    Traverses two different lists in parallel and combines their elements using 'fn'.
    For example, fn=add is used to sum two lists.

    Args:
        fn: A function that combines two values.

    Returns:
         A function that takes two lists of the same size and produces a new list by applying fn(x, y).
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return lambda ls1, ls2: (fn(x, y) for x, y in zip(ls1, ls2))


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add corresponding elements of two lists using zipWith and add.
    Combines two lists like a zipper using the 'add' operation.
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-Order Function: Reduce

    Used to decrease ("reduce") a list to a single value.
    For example, to find the sum: start=0, fn=add.
    The order of operation is: $fn(x_3, fn(x_2, fn(x_1, start)))$

    Args:
        fn: A function that combines two values.
        start: Starting value $x_0$.

    Returns:
         A function that takes a list and returns a single reduced value.
    """

    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    def reducer(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(x, result)
        return result

    return reducer


def sum(ls: Iterable[float]) -> float:
    """
    Sum all elements in a list using reduce and add.
    Uses 'reduce' and 'add' to find the total sum.
    The starting value (start) should be 0.
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Calculate the product of all elements in a list using reduce and mul.
    Uses 'reduce' and 'mul' to find the total product.
    The starting value (start) should be 1.
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return reduce(mul, 1)(ls)
