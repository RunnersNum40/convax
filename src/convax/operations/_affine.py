from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp

from convax._arrays import as_float_array, require_matrix, require_vector_dimension
from convax._types import IntegerVectorLike, MatrixLike, VectorLike
from convax.sets import (
    AbstractConvexSet,
    AbstractSupportSet,
    AffineImage,
    ConstrainedZonotope,
    HalfspacePolyhedron,
)


@overload
def affine_map(
    convex_set: ConstrainedZonotope,
    matrix: MatrixLike,
    offset: VectorLike | None = None,
) -> ConstrainedZonotope: ...


@overload
def affine_map(
    convex_set: AbstractSupportSet,
    matrix: MatrixLike,
    offset: VectorLike | None = None,
) -> AffineImage: ...


def affine_map(
    convex_set: AbstractConvexSet,
    matrix: MatrixLike,
    offset: VectorLike | None = None,
) -> ConstrainedZonotope | AffineImage:
    """Return an affine image while preserving supported representations."""
    if isinstance(convex_set, ConstrainedZonotope):
        matrix = as_float_array(matrix)
        require_matrix("matrix", matrix)
        if matrix.shape[1] != convex_set.ambient_dimension:
            raise ValueError(
                "matrix columns must match the set dimension, got "
                f"{matrix.shape} and {convex_set.ambient_dimension}"
            )
        if offset is None:
            offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
        else:
            offset = as_float_array(offset)
        require_vector_dimension("offset", offset, matrix.shape[0])
        dtype = jnp.result_type(convex_set.dtype, matrix.dtype, offset.dtype)
        matrix = matrix.astype(dtype)
        offset = offset.astype(dtype)
        center = convex_set.center.astype(dtype)
        generator_matrix = convex_set.generator_matrix.astype(dtype)
        return ConstrainedZonotope(
            matrix @ center + offset,
            matrix @ generator_matrix,
            convex_set.constraint_matrix,
            convex_set.constraint_values,
        )
    if isinstance(convex_set, AbstractSupportSet):
        return AffineImage(convex_set, matrix, offset)
    raise TypeError(f"affine map is not implemented for {type(convex_set).__name__}")


def affine_preimage(
    polyhedron: HalfspacePolyhedron,
    matrix: MatrixLike,
    offset: VectorLike | None = None,
) -> HalfspacePolyhedron:
    r"""Return :math:`\{x \mid Ax + b \in P\}` for a halfspace polyhedron."""
    matrix = as_float_array(matrix)
    require_matrix("matrix", matrix)
    if matrix.shape[0] != polyhedron.ambient_dimension:
        raise ValueError(
            "matrix rows must match the polyhedron dimension, got "
            f"{matrix.shape} and {polyhedron.ambient_dimension}"
        )
    if offset is None:
        offset = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)
    else:
        offset = as_float_array(offset)
    require_vector_dimension("offset", offset, matrix.shape[0])
    dtype = jnp.result_type(polyhedron.dtype, matrix.dtype, offset.dtype)
    matrix = matrix.astype(dtype)
    offset = offset.astype(dtype)
    inequality_matrix = polyhedron.inequality_matrix.astype(dtype)
    inequality_bounds = polyhedron.inequality_bounds.astype(dtype)
    equality_matrix = polyhedron.equality_matrix.astype(dtype)
    equality_values = polyhedron.equality_values.astype(dtype)
    return HalfspacePolyhedron(
        inequality_matrix @ matrix,
        inequality_bounds - inequality_matrix @ offset,
        equality_matrix @ matrix,
        equality_values - equality_matrix @ offset,
    )


@overload
def project_coordinates(
    convex_set: ConstrainedZonotope,
    coordinates: IntegerVectorLike,
) -> ConstrainedZonotope: ...


@overload
def project_coordinates(
    convex_set: AbstractSupportSet,
    coordinates: IntegerVectorLike,
) -> AffineImage: ...


def project_coordinates(
    convex_set: AbstractConvexSet,
    coordinates: IntegerVectorLike,
) -> ConstrainedZonotope | AffineImage:
    """Project a set onto an ordered collection of coordinate axes."""
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
    if isinstance(convex_set, ConstrainedZonotope):
        return ConstrainedZonotope(
            convex_set.center[coordinates],
            convex_set.generator_matrix[coordinates],
            convex_set.constraint_matrix,
            convex_set.constraint_values,
        )
    if not isinstance(convex_set, AbstractSupportSet):
        raise TypeError(
            f"coordinate projection is not implemented for {type(convex_set).__name__}"
        )
    projection_matrix = jax.nn.one_hot(
        coordinates,
        convex_set.ambient_dimension,
        dtype=convex_set.dtype,
    )
    return AffineImage(convex_set, projection_matrix)


@overload
def translate(
    convex_set: ConstrainedZonotope,
    offset: VectorLike,
) -> ConstrainedZonotope: ...


@overload
def translate(
    convex_set: AbstractSupportSet,
    offset: VectorLike,
) -> AffineImage: ...


@overload
def translate(
    convex_set: HalfspacePolyhedron,
    offset: VectorLike,
) -> HalfspacePolyhedron: ...


def translate(
    convex_set: AbstractConvexSet,
    offset: VectorLike,
) -> AbstractConvexSet:
    """Translate a set by an ambient-space offset."""
    offset = as_float_array(offset)
    require_vector_dimension("offset", offset, convex_set.ambient_dimension)
    if isinstance(convex_set, ConstrainedZonotope):
        dtype = jnp.result_type(convex_set.dtype, offset.dtype)
        return ConstrainedZonotope(
            convex_set.center.astype(dtype) + offset.astype(dtype),
            convex_set.generator_matrix,
            convex_set.constraint_matrix,
            convex_set.constraint_values,
        )
    if isinstance(convex_set, HalfspacePolyhedron):
        return HalfspacePolyhedron(
            convex_set.inequality_matrix,
            convex_set.inequality_bounds + convex_set.inequality_matrix @ offset,
            convex_set.equality_matrix,
            convex_set.equality_values + convex_set.equality_matrix @ offset,
        )
    if isinstance(convex_set, AbstractSupportSet):
        return AffineImage(
            convex_set,
            jnp.eye(convex_set.ambient_dimension, dtype=convex_set.dtype),
            offset,
        )
    raise TypeError(f"translation is not implemented for {type(convex_set).__name__}")


@overload
def negate(convex_set: ConstrainedZonotope) -> ConstrainedZonotope: ...


@overload
def negate(convex_set: AbstractSupportSet) -> AffineImage: ...


@overload
def negate(convex_set: HalfspacePolyhedron) -> HalfspacePolyhedron: ...


def negate(convex_set: AbstractConvexSet) -> AbstractConvexSet:
    """Reflect a set through the origin."""
    if isinstance(convex_set, ConstrainedZonotope):
        return ConstrainedZonotope(
            -convex_set.center,
            -convex_set.generator_matrix,
            convex_set.constraint_matrix,
            convex_set.constraint_values,
        )
    if isinstance(convex_set, HalfspacePolyhedron):
        return HalfspacePolyhedron(
            -convex_set.inequality_matrix,
            convex_set.inequality_bounds,
            -convex_set.equality_matrix,
            convex_set.equality_values,
        )
    if isinstance(convex_set, AbstractSupportSet):
        return AffineImage(
            convex_set,
            -jnp.eye(convex_set.ambient_dimension, dtype=convex_set.dtype),
        )
    raise TypeError(f"negation is not implemented for {type(convex_set).__name__}")
