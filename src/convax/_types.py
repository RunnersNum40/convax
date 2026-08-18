from collections.abc import Sequence

from jaxtyping import ArrayLike

type VectorLike = ArrayLike | Sequence[float | int]
type MatrixLike = ArrayLike | Sequence[Sequence[float | int]]
type IntegerVectorLike = ArrayLike | Sequence[int]
