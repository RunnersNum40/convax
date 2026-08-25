from collections.abc import Sequence
from typing import final, override

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Bool, Float, Real, ScalarLike

from convax._utils import (
    _affine_map_center_and_generator_matrix,
    _scaled_l2_norm,
    normalize_center_and_generator_matrix,
    normalize_query_vector,
    normalize_tolerance,
)
from convax.sets._abstract import (
    AbstractAffineMapSet,
    AbstractPointContainmentSet,
    AbstractSupportSet,
)
from convax.sets._results import SupportResult


@final
class Ellipsoid(
    AbstractAffineMapSet,
    AbstractSupportSet,
    AbstractPointContainmentSet,
):
    r"""Affine image of a Euclidean unit ball.

    Represents \(\{c + Gu \mid \lVert u \rVert_2 \leq 1\}\). The generator
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
        center: Real[ArrayLike, "ambient_dimension"] | Sequence[float | int],
        generator_matrix: Real[ArrayLike, "ambient_dimension latent_dimension"]
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
    ) -> "Ellipsoid":
        center, generator_matrix = _affine_map_center_and_generator_matrix(
            self.center,
            self.generator_matrix,
            matrix,
            offset,
            source_dtype=self.dtype,
        )
        return Ellipsoid(center, generator_matrix)

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

        center = self.center.astype(direction.dtype)
        generator_matrix = self.generator_matrix.astype(direction.dtype)

        latent_direction = generator_matrix.T @ direction
        latent_norm = _scaled_l2_norm(latent_direction)

        def nonzero_support_point() -> Float[Array, "ambient_dimension"]:
            return self.center + self.generator_matrix @ (
                latent_direction / latent_norm
            )

        point = jax.lax.cond(
            latent_norm > 0,
            nonzero_support_point,
            lambda: center,
        )
        value = center @ direction + latent_norm
        return SupportResult(value=value, point=point)

    @override
    def contains(
        self,
        point: Real[ArrayLike, "{self.ambient_dimension}"] | Sequence[float | int],
        *,
        tolerance: ScalarLike = 1e-6,
    ) -> Bool[Array, ""]:
        """Return whether a point belongs to the ellipsoid.

        Args:
            point: Query point with shape ``(ambient_dimension,)``.
            tolerance: Finite, nonnegative scalar feasibility tolerance.
        """
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
            return _scaled_l2_norm(displacement) <= tolerance
        latent_point = jnp.linalg.pinv(generator_matrix) @ displacement
        reconstruction_error = _scaled_l2_norm(
            generator_matrix @ latent_point - displacement
        )
        displacement_scale = jnp.maximum(
            _scaled_l2_norm(displacement), jnp.asarray(1, dtype=dtype)
        )
        in_affine_hull = reconstruction_error <= tolerance * displacement_scale
        in_unit_ball = _scaled_l2_norm(latent_point) <= 1 + tolerance
        return jnp.logical_and(in_affine_hull, in_unit_ball)
