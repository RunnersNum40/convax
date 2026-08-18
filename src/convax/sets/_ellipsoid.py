from typing import final, override

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, ScalarLike

from convax._types import MatrixLike, VectorLike
from convax.sets._abstract import (
    AbstractPointContainmentSet,
    AbstractSupportSet,
    normalize_center_and_generator_matrix,
    normalize_query_vector,
    normalize_tolerance,
)
from convax.sets._results import SupportResult


@final
class Ellipsoid(AbstractSupportSet, AbstractPointContainmentSet):
    r"""An affine image of a Euclidean unit ball.

    Represents :math:`\{c + Gu \mid \lVert u \rVert_2 \leq 1\}`. The generator
    matrix may be rectangular or rank deficient.

    Args:
        center: Ellipsoid center with shape ``(ambient_dimension,)``.
        generator_matrix: Generator matrix with shape
            ``(ambient_dimension, latent_dimension)``.
    """

    center: Float[Array, "ambient_dimension"]
    generator_matrix: Float[Array, "ambient_dimension latent_dimension"]

    def __init__(
        self,
        center: VectorLike,
        generator_matrix: MatrixLike,
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
    def support(self, direction: VectorLike) -> SupportResult:
        direction = normalize_query_vector(
            "direction", direction, self.ambient_dimension, dtype=self.dtype
        )
        latent_direction = self.generator_matrix.T @ direction
        latent_norm = jnp.linalg.norm(latent_direction)

        def nonzero_support_point() -> Float[Array, "ambient_dimension"]:
            return self.center + self.generator_matrix @ (
                latent_direction / latent_norm
            )

        point = jax.lax.cond(
            latent_norm > 0,
            nonzero_support_point,
            lambda: self.center,
        )
        value = self.center @ direction + latent_norm
        return SupportResult(value=value, point=point)

    @override
    def contains(
        self,
        point: VectorLike,
        *,
        tolerance: ScalarLike = 1e-6,
    ) -> Bool[Array, ""]:
        point = normalize_query_vector(
            "point", point, self.ambient_dimension, dtype=self.dtype
        )
        tolerance = normalize_tolerance(tolerance, dtype=self.dtype)
        dtype = jnp.result_type(self.dtype, point.dtype, tolerance.dtype)
        point = point.astype(dtype)
        tolerance = tolerance.astype(dtype)
        center = self.center.astype(dtype)
        generator_matrix = self.generator_matrix.astype(dtype)
        displacement = point - center
        if generator_matrix.shape[1] == 0:
            return jnp.linalg.norm(displacement) <= tolerance
        latent_point = jnp.linalg.pinv(generator_matrix) @ displacement
        reconstruction_error = jnp.linalg.norm(
            generator_matrix @ latent_point - displacement
        )
        displacement_scale = jnp.maximum(
            jnp.linalg.norm(displacement), jnp.asarray(1, dtype=dtype)
        )
        in_affine_hull = reconstruction_error <= tolerance * displacement_scale
        in_unit_ball = jnp.linalg.norm(latent_point) <= 1 + tolerance
        return jnp.logical_and(in_affine_hull, in_unit_ball)
