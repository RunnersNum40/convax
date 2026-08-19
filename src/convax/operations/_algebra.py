from typing import overload

import jax.numpy as jnp
from jax.scipy.linalg import block_diag as block_diagonal

from convax.sets import (
    AbstractConvexSet,
    AbstractSupportSet,
    ConstrainedZonotope,
    ConvexHull,
    HalfspacePolyhedron,
    MinkowskiSum,
)


def convex_hull(
    left_set: AbstractSupportSet, right_set: AbstractSupportSet
) -> ConvexHull:
    """Return the convex hull of two compact convex sets."""
    return ConvexHull(left_set, right_set)


@overload
def intersection(
    left_set: HalfspacePolyhedron, right_set: HalfspacePolyhedron
) -> HalfspacePolyhedron: ...


@overload
def intersection(
    left_set: ConstrainedZonotope, right_set: ConstrainedZonotope
) -> ConstrainedZonotope: ...


def intersection(
    left_set: AbstractConvexSet, right_set: AbstractConvexSet
) -> HalfspacePolyhedron | ConstrainedZonotope:
    """Intersect matching halfspace or constrained-zonotope representations."""
    if isinstance(left_set, HalfspacePolyhedron) and isinstance(
        right_set, HalfspacePolyhedron
    ):
        if left_set.ambient_dimension != right_set.ambient_dimension:
            raise ValueError(
                "intersection dimensions must match, got "
                f"{left_set.ambient_dimension} and {right_set.ambient_dimension}"
            )
        return HalfspacePolyhedron(
            jnp.concatenate(
                (left_set.inequality_matrix, right_set.inequality_matrix), axis=0
            ),
            jnp.concatenate(
                (left_set.inequality_bounds, right_set.inequality_bounds), axis=0
            ),
            jnp.concatenate(
                (left_set.equality_matrix, right_set.equality_matrix), axis=0
            ),
            jnp.concatenate(
                (left_set.equality_values, right_set.equality_values), axis=0
            ),
        )
    if isinstance(left_set, ConstrainedZonotope) and isinstance(
        right_set, ConstrainedZonotope
    ):
        if left_set.ambient_dimension != right_set.ambient_dimension:
            raise ValueError(
                "intersection dimensions must match, got "
                f"{left_set.ambient_dimension} and {right_set.ambient_dimension}"
            )
        dtype = jnp.result_type(left_set.dtype, right_set.dtype)
        left_center = left_set.center.astype(dtype)
        right_center = right_set.center.astype(dtype)
        left_generators = left_set.generator_matrix.astype(dtype)
        right_generators = right_set.generator_matrix.astype(dtype)
        left_constraints = left_set.constraint_matrix.astype(dtype)
        right_constraints = right_set.constraint_matrix.astype(dtype)
        output_generators = jnp.concatenate(
            (left_generators, jnp.zeros_like(right_generators)),
            axis=1,
        )
        operand_constraints = block_diagonal(left_constraints, right_constraints)
        matching_constraints = jnp.concatenate(
            (left_generators, -right_generators), axis=1
        )
        output_constraints = jnp.concatenate(
            (operand_constraints, matching_constraints),
            axis=0,
        )
        output_constraint_values = jnp.concatenate(
            (
                left_set.constraint_values.astype(dtype),
                right_set.constraint_values.astype(dtype),
                right_center - left_center,
            )
        )
        return ConstrainedZonotope(
            left_center,
            output_generators,
            output_constraints,
            output_constraint_values,
        )
    raise TypeError(
        "intersection is not implemented for "
        f"{type(left_set).__name__} and {type(right_set).__name__}"
    )


@overload
def minkowski_sum(
    left_set: AbstractSupportSet, right_set: AbstractSupportSet
) -> MinkowskiSum: ...


@overload
def minkowski_sum(
    left_set: ConstrainedZonotope, right_set: ConstrainedZonotope
) -> ConstrainedZonotope: ...


def minkowski_sum(
    left_set: AbstractConvexSet, right_set: AbstractConvexSet
) -> MinkowskiSum | ConstrainedZonotope:
    """Return a Minkowski sum while preserving supported representations."""
    if isinstance(left_set, ConstrainedZonotope) and isinstance(
        right_set, ConstrainedZonotope
    ):
        if left_set.ambient_dimension != right_set.ambient_dimension:
            raise ValueError(
                "Minkowski sum dimensions must match, got "
                f"{left_set.ambient_dimension} and {right_set.ambient_dimension}"
            )
        dtype = jnp.result_type(left_set.dtype, right_set.dtype)
        left_generators = left_set.generator_matrix.astype(dtype)
        right_generators = right_set.generator_matrix.astype(dtype)
        left_constraints = left_set.constraint_matrix.astype(dtype)
        right_constraints = right_set.constraint_matrix.astype(dtype)
        output_constraints = block_diagonal(left_constraints, right_constraints)
        return ConstrainedZonotope(
            left_set.center.astype(dtype) + right_set.center.astype(dtype),
            jnp.concatenate((left_generators, right_generators), axis=1),
            output_constraints,
            jnp.concatenate(
                (
                    left_set.constraint_values.astype(dtype),
                    right_set.constraint_values.astype(dtype),
                )
            ),
        )
    if isinstance(left_set, AbstractSupportSet) and isinstance(
        right_set, AbstractSupportSet
    ):
        return MinkowskiSum(left_set, right_set)
    raise TypeError(
        "Minkowski sum is not implemented for "
        f"{type(left_set).__name__} and {type(right_set).__name__}"
    )
