from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError('Need to implement for Task 1.1')
    vals_list: List[float] = list(vals)

    # İlgilendiğimiz değişkenin orijinal değerini saklayalım (güvenlik için)
    original_val = vals_list[arg]

    # 1. Adım: f(x + epsilon) hesapla
    vals_list[arg] = original_val + epsilon
    # * operatörü ile listeyi argümanlara açarak fonksiyonu çağırıyoruz
    f_plus = f(*vals_list)

    # 2. Adım: f(x - epsilon) hesapla
    vals_list[arg] = original_val - epsilon
    f_minus = f(*vals_list)

    # Listeyi orijinal haline döndürelim (iyi bir mühendislik pratiği, yan etki bırakmamak için)
    vals_list[arg] = original_val

    # 3. Adım: Merkezi fark formülü
    # Türev = (f(x+h) - f(x-h)) / 2h
    derivative = (f_plus - f_minus) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")

    order = []
    visited = set()

    def visit(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            visit(parent)
        order.append(v)

    visit(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")

    # Adım 0: Türevleri saklamak için bir sözlük (dictionary) oluştur.
    # Bu sözlük, her bir değişkenin (Variable) unique_id'si ile o anki hesaplanmış türevini eşleştirecek.
    derivatives = {}

    # Başlangıç noktası: En tepedeki (right-most) değişkenin türevi bize veriliyor.
    # Bunu sözlüğe ekleyerek başlıyoruz.
    derivatives[variable.unique_id] = deriv

    # Adım 1: Hesaplama grafiğindeki düğümleri topolojik sıraya diz.
    # Bu fonksiyon bize değişkenleri, sonuçtan sebebe (output -> input) doğru giden bir sırada verir.
    # Bu sıra önemlidir çünkü bir düğümün türevini hesaplamadan önce, onu kullanan tüm düğümlerin
    # türevlerinin hesaplanmış ve toplanmış olması gerekir.
    order = topological_sort(variable)

    # Adım 2: Sıralanmış değişkenler üzerinde döngü başlat.
    for var in order:
        # Şu anki değişken için birikmiş türevi al.
        d_output = derivatives.get(var.unique_id)

        # Eğer bu değişken bir "yaprak" (leaf) ise:
        # Bu, işlemin en başındaki girdilerden biri olduğu anlamına gelir (örn. kullanıcı tarafından oluşturulan w veya x).
        # Bu durumda, hesapladığımız türevi bu değişkene "biriktirmemiz" (accumulate) gerekir.
        if var.is_leaf():
            var.accumulate_derivative(d_output)

        # Eğer yaprak değilse (yani bir ara işlem sonucuysa):
        # Zincir kuralını (chain rule) uygulayarak türevi bu değişkenin ebeveynlerine (parents) dağıtmalıyız.
        else:
            # chain_rule fonksiyonu, bu değişkenin türevini (d_output) alır ve
            # bunu oluşturan girdilere (parents) ne kadar türev gitmesi gerektiğini hesaplar.
            # Bize (parent_variable, derivative_part) çiftleri döndürür.
            for parent, d_parent in var.chain_rule(d_output):
                # Ebeveynin türevini sözlüğümüze ekle veya varsa üzerine topla.
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += d_parent
                else:
                    derivatives[parent.unique_id] = d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
