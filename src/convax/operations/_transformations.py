from collections.abc import Sequence
from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer, Real

from convax.sets import (
    AbstractAffineMapSet,
    AbstractAffinePreimageSet,
    AbstractConvexSet,
    AbstractNegationSet,
    AbstractSupportSet,
    AbstractTranslationSet,
    AffineImage,
)


def _affine_map_or_image(
    convex_set: AbstractConvexSet,
    matrix: Real[ArrayLike, "_ _"] | Sequence[Sequence[float | int]],
    offset: Real[ArrayLike, "_"] | Sequence[float | int] | None,
    operation_name: str,
) -> AbstractConvexSet:
    if isinstance(convex_set, AbstractAffineMapSet):
        return convex_set._affine_map(matrix, offset)
    if isinstance(convex_set, AbstractSupportSet):
        return AffineImage(convex_set, matrix, offset)
    raise TypeError(
        f"{operation_name} is not implemented for {type(convex_set).__name__}"
    )


@overload
def affine_map[SetT: AbstractAffineMapSet](
    convex_set: SetT,
    matrix: Real[ArrayLike, "output_dimension {convex_set.ambient_dimension}"]
    | Sequence[Sequence[float | int]],
    offset: Real[ArrayLike, "output_dimension"] | Sequence[float | int] | None = None,
) -> SetT: ...


@overload
def affine_map(
    convex_set: AbstractSupportSet,
    matrix: Real[ArrayLike, "output_dimension {convex_set.ambient_dimension}"]
    | Sequence[Sequence[float | int]],
    offset: Real[ArrayLike, "output_dimension"] | Sequence[float | int] | None = None,
) -> AbstractSupportSet: ...


def affine_map(
    convex_set: AbstractConvexSet,
    matrix: Real[ArrayLike, "output_dimension {convex_set.ambient_dimension}"]
    | Sequence[Sequence[float | int]],
    offset: Real[ArrayLike, "output_dimension"] | Sequence[float | int] | None = None,
) -> AbstractConvexSet:
    """Return the affine image of a convex set.

    Retain affine-map-closed representations; otherwise, return an exact lazy ``AffineImage``
    for support-capable sets.

    Args:
        convex_set: Convex set to transform.
        matrix: Linear-map matrix with shape
            ``(output_dimension, convex_set.ambient_dimension)``.
        offset: Optional translation vector with shape ``(output_dimension,)``.
            ``None`` selects a zero offset.

    Raises:
        TypeError: If neither a representation-preserving construction nor the
            support-function fallback is available.
        ValueError: If an input has invalid rank or dimension.
    """
    return _affine_map_or_image(convex_set, matrix, offset, "affine map")


def affine_preimage[SetT: AbstractAffinePreimageSet](
    convex_set: SetT,
    matrix: Real[ArrayLike, "{convex_set.ambient_dimension} input_dimension"]
    | Sequence[Sequence[float | int]],
    offset: Real[ArrayLike, "{convex_set.ambient_dimension}"]
    | Sequence[float | int]
    | None = None,
) -> SetT:
    """Return the affine preimage of a convex set in the same representation.

    Args:
        convex_set: Affine-preimage-closed convex set.
        matrix: Linear-map matrix with shape
            ``(convex_set.ambient_dimension, input_dimension)``.
        offset: Optional vector with shape ``(convex_set.ambient_dimension,)``
            added before membership testing. ``None`` selects a zero offset.

    Raises:
        TypeError: If the representation is not affine-preimage-closed or an input contains
            complex values.
        ValueError: If an input has invalid rank or dimension.
    """
    if not isinstance(convex_set, AbstractAffinePreimageSet):
        raise TypeError(
            f"affine preimage is not implemented for {type(convex_set).__name__}"
        )
    return convex_set._affine_preimage(matrix, offset)


@overload
def project_coordinates[SetT: AbstractAffineMapSet](
    convex_set: SetT,
    coordinates: Integer[ArrayLike, "output_dimension"] | Sequence[int],
) -> SetT: ...


@overload
def project_coordinates(
    convex_set: AbstractSupportSet,
    coordinates: Integer[ArrayLike, "output_dimension"] | Sequence[int],
) -> AbstractSupportSet: ...


def project_coordinates(
    convex_set: AbstractConvexSet,
    coordinates: Integer[ArrayLike, "output_dimension"] | Sequence[int],
) -> AbstractConvexSet:
    """Project a convex set onto ordered coordinate axes.

    Args:
        convex_set: Convex set to project.
        coordinates: One-dimensional integer sequence of coordinate indices. Its
            order determines the output order.

    Raises:
        TypeError: If coordinates are not integers or no exact construction is
            available.
        ValueError: If coordinates are not one-dimensional.
        EquinoxRuntimeError: If an index lies outside the ambient dimension.
    """
    coordinates = jnp.asarray(coordinates)
    if coordinates.ndim != 1:
        raise ValueError(f"coordinates must be a vector, got shape {coordinates.shape}")
    if not jnp.issubdtype(coordinates.dtype, jnp.integer):
        raise TypeError("coordinates must contain integers")
    coordinates = eqx.error_if(
        coordinates,
        jnp.any((coordinates < 0) | (coordinates >= convex_set.ambient_dimension)),
        "coordinates must lie within the set's ambient dimension",
    )
    projection_matrix = jax.nn.one_hot(
        coordinates,
        convex_set.ambient_dimension,
        dtype=convex_set.dtype,
    )
    return _affine_map_or_image(
        convex_set,
        projection_matrix,
        None,
        "coordinate projection",
    )


@overload
def translate[SetT: AbstractTranslationSet](
    convex_set: SetT,
    offset: Real[ArrayLike, "{convex_set.ambient_dimension}"] | Sequence[float | int],
) -> SetT: ...


@overload
def translate(
    convex_set: AbstractSupportSet,
    offset: Real[ArrayLike, "{convex_set.ambient_dimension}"] | Sequence[float | int],
) -> AbstractSupportSet: ...


def translate(
    convex_set: AbstractConvexSet,
    offset: Real[ArrayLike, "{convex_set.ambient_dimension}"] | Sequence[float | int],
) -> AbstractConvexSet:
    """Return a translated convex set.

    Args:
        convex_set: Convex set to translate.
        offset: Translation vector with shape ``(convex_set.ambient_dimension,)``.

    Raises:
        TypeError: If no exact construction is available.
        ValueError: If the offset has invalid rank or dimension.
    """
    if isinstance(convex_set, AbstractTranslationSet):
        return convex_set._translate(offset)
    if isinstance(convex_set, AbstractSupportSet):
        return AffineImage(
            convex_set,
            jnp.eye(convex_set.ambient_dimension, dtype=convex_set.dtype),
            offset,
        )
    raise TypeError(f"translation is not implemented for {type(convex_set).__name__}")


@overload
def negate[SetT: AbstractNegationSet](convex_set: SetT) -> SetT: ...


@overload
def negate(convex_set: AbstractSupportSet) -> AbstractSupportSet: ...


def negate(convex_set: AbstractConvexSet) -> AbstractConvexSet:
    """Return a convex set reflected through the origin.

    Args:
        convex_set: Convex set to negate.

    Raises:
        TypeError: If no exact construction is available.
    """
    if isinstance(convex_set, AbstractNegationSet):
        return convex_set._negate()
    if isinstance(convex_set, AbstractSupportSet):
        return AffineImage(
            convex_set,
            -jnp.eye(convex_set.ambient_dimension, dtype=convex_set.dtype),
        )
    raise TypeError(f"negation is not implemented for {type(convex_set).__name__}")
