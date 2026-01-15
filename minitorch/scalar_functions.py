from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Çarpma işleminin türevi için zincir kuralını uygular.
        f(a, b) = a * b
        d_out/da = b
        d_out/db = a
        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        ctx.save_for_backward(a)
        return 1.0 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Ters alma işleminin türevi: f(x) = 1/x -> f'(x) = -1/x^2
        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Negatif alma işleminin türevi sabittir (-1).
        f(x) = -x -> f'(x) = -1
        """
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        s = operators.sigmoid(a)
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Sigmoid fonksiyonunun türevi: f'(x) = f(x) * (1 - f(x))
        Burada kayıtlı değer zaten f(x)'tir.
        """
        sigma = ctx.saved_values[0]
        return d_output * sigma * (1.0 - sigma)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Geriye yayılım (Backpropagation) işlemi:
        ReLU fonksiyonunun türevi: Eğer x > 0 ise 1, değilse 0.
        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        exp_a = operators.exp(a)
        ctx.save_for_backward(exp_a)
        return exp_a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Üstel fonksiyonun türevi kendisidir: f(x) = e^x -> f'(x) = e^x
        Kayıtlı değer zaten e^x sonucudur.
        """
        out = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Küçüktür (<) işlemi bir karşılaştırma olduğundan türevi her yerde 0'dır (veya tanımsızdır).
        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        # raise NotImplementedError('Need to implement for Task 1.2')
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Geriye yayılım (Backpropagation) işlemi:
        Eşittir (==) işlemi bir karşılaştırma olduğundan türevi her yerde 0'dır.
        """
        return 0.0, 0.0
