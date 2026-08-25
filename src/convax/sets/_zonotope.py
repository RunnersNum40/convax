from collections.abc import Sequence
from typing import final, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float, Real

from convax._utils import (
    _affine_map_center_and_generator_matrix,
    normalize_center_and_generator_matrix,
    normalize_query_vector,
)
from convax.sets._abstract import (
    AbstractAffineMapSet,
    AbstractSupportSet,
)
from convax.sets._results import SupportResult


@final
class Zonotope(AbstractAffineMapSet, AbstractSupportSet):
    r"""Affine image of a unit infinity-norm ball.

    Represents \(\{c + G\xi \mid \lVert \xi \rVert_\infty \leq 1\}\).

    Args:
        center: Zonotope center with shape ``(ambient_dimension,)``.
        generator_matrix: Generator matrix with shape
            ``(ambient_dimension, generator_count)``.
    """

    center: Float[Array, "ambient_dimension"]
    generator_matrix: Float[Array, "ambient_dimension generator_count"]

    def __init__(
        self,
        center: Real[ArrayLike, "ambient_dimension"] | Sequence[float | int],
        generator_matrix: Real[ArrayLike, "ambient_dimension generator_count"]
        | Sequence[Sequence[float | int]],
    ) -> None:
        self.center, self.generator_matrix = normalize_center_and_generator_matrix(
            center, generator_matrix
        )

    @property
    def ambient_dimension(self) -> int:
        return self.center.shape[0]

    @property
    def dtype(self):
        return self.center.dtype

    @override
    def _affine_map(
        self,
        matrix: Real[ArrayLike, "output_dimension {self.ambient_dimension}"]
        | Sequence[Sequence[float | int]],
        offset: Real[ArrayLike, "output_dimension"]
        | Sequence[float | int]
        | None = None,
    ) -> "Zonotope":
        center, generator_matrix = _affine_map_center_and_generator_matrix(
            self.center,
            self.generator_matrix,
            matrix,
            offset,
            source_dtype=self.dtype,
        )
        return Zonotope(center, generator_matrix)

    @override
    def support(
        self,
        direction: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
    ) -> SupportResult:
        """Return the support value and a maximizing point.

        Args:
            direction: Support-query vector with shape ``(ambient_dimension,)``.
        """
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        latent_direction = self.generator_matrix.T @ direction
        latent_point = jnp.sign(latent_direction)
        point = self.center + self.generator_matrix @ latent_point
        value = self.center @ direction + jnp.sum(jnp.abs(latent_direction))
        return SupportResult(value=value, point=point)
