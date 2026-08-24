from typing import Any, assert_type, cast

import jax
import jax.numpy as jnp
import pytest

from convax import (
    ConstrainedZonotope,
    Zonotope,
    intersection,
    minkowski_sum,
)


def constrained_zonotope(
    center: jax.Array | list[int],
) -> ConstrainedZonotope:
    return ConstrainedZonotope(
        center,
        [[1, 0], [0, 2]],
        [[1, -1]],
        [0],
    )


def test_affine_map_preserves_constrained_zonotope_representation() -> None:
    source = constrained_zonotope([1, -1])
    matrix = jnp.array([[2.0, 0.0], [1.0, -1.0]])
    offset = jnp.array([0.5, 2.0])

    eager = source.affine_map(matrix, offset)
    compiled = jax.jit(
        lambda convex_set, affine_matrix, affine_offset: convex_set.affine_map(
            affine_matrix, affine_offset
        )
    )(source, matrix, offset)

    assert jnp.allclose(eager.center, matrix @ source.center + offset)
    assert jnp.allclose(eager.generator_matrix, matrix @ source.generator_matrix)
    assert jnp.array_equal(eager.constraint_matrix, source.constraint_matrix)
    assert jnp.array_equal(eager.constraint_values, source.constraint_values)
    assert jnp.array_equal(compiled.center, eager.center)
    assert jnp.array_equal(compiled.generator_matrix, eager.generator_matrix)


def test_affine_map_promotes_before_arithmetic() -> None:
    source = ConstrainedZonotope(
        jnp.array([1], dtype=jnp.float16),
        jnp.ones((1, 1), dtype=jnp.float16),
        jnp.empty((0, 1), dtype=jnp.float16),
        jnp.empty((0,), dtype=jnp.float16),
    )
    matrix = jnp.array([[1.0004]], dtype=jnp.float32)
    offset = jnp.array([0.0004], dtype=jnp.float16)
    expected = matrix @ source.center.astype(jnp.float32) + offset.astype(jnp.float32)

    image = source.affine_map(matrix, offset)

    assert image.dtype == jnp.float32
    assert jnp.array_equal(image.center, expected)


def test_coordinate_projection_preserves_order_constraints_and_duplicates() -> None:
    source = ConstrainedZonotope(
        [1, 2, 3],
        [[1, 0], [0, 1], [2, 3]],
        [[1, -1]],
        [0],
    )
    coordinates = jnp.array([2, 0, 2])

    projected = jax.jit(
        lambda convex_set, selected_coordinates: convex_set.project_coordinates(
            selected_coordinates
        )
    )(source, coordinates)

    assert jnp.array_equal(projected.center, jnp.array([3.0, 1.0, 3.0]))
    assert jnp.array_equal(
        projected.generator_matrix,
        jnp.array([[2.0, 3.0], [1.0, 0.0], [2.0, 3.0]]),
    )
    assert jnp.array_equal(projected.constraint_matrix, source.constraint_matrix)
    assert jnp.array_equal(projected.constraint_values, source.constraint_values)


def test_zero_dimensional_operations_are_jittable_and_vectorizable() -> None:
    centers = jnp.empty((2, 0))
    generator_matrix = jnp.empty((0, 1))
    constraint_matrix = jnp.empty((0, 1))
    constraint_values = jnp.empty((0,))
    coordinates = jnp.empty((0,), dtype=jnp.int32)
    source = ConstrainedZonotope(
        centers[0],
        generator_matrix,
        constraint_matrix,
        constraint_values,
    )

    eager_projection = source.project_coordinates(coordinates)
    compiled_projection = jax.jit(
        lambda convex_set, selected_coordinates: convex_set.project_coordinates(
            selected_coordinates
        )
    )(source, coordinates)
    batched_sets = jax.vmap(
        lambda center: ConstrainedZonotope(
            center,
            generator_matrix,
            constraint_matrix,
            constraint_values,
        )
    )(centers)
    batched_projections = jax.jit(
        jax.vmap(lambda convex_set: convex_set.project_coordinates(coordinates))
    )(batched_sets)
    summed = jax.jit(minkowski_sum)(source, source)
    intersected = jax.jit(intersection)(source, source)

    assert eager_projection.center.shape == (0,)
    assert eager_projection.generator_matrix.shape == (0, 1)
    assert compiled_projection.generator_matrix.shape == (0, 1)
    assert batched_projections.center.shape == (2, 0)
    assert batched_projections.generator_matrix.shape == (2, 0, 1)
    assert summed.center.shape == (0,)
    assert summed.generator_matrix.shape == (0, 2)
    assert intersected.center.shape == (0,)
    assert intersected.constraint_matrix.shape == (0, 2)


def test_translation_and_negation_preserve_latent_constraints() -> None:
    source = constrained_zonotope([1, -1])
    offset = jnp.array([2.0, 3.0])

    translated = source.translate(offset)
    reflected = source.negate()

    assert jnp.array_equal(translated.center, source.center + offset)
    assert jnp.array_equal(translated.generator_matrix, source.generator_matrix)
    assert jnp.array_equal(translated.constraint_matrix, source.constraint_matrix)
    assert jnp.array_equal(reflected.center, -source.center)
    assert jnp.array_equal(reflected.generator_matrix, -source.generator_matrix)
    assert jnp.array_equal(reflected.constraint_values, source.constraint_values)


def test_minkowski_sum_builds_block_diagonal_constraints() -> None:
    left = ConstrainedZonotope([1], [[2]], [[1]], [0.5])
    right = ConstrainedZonotope([-2], [[3, 4]], [[1, -1]], [0])

    eager = minkowski_sum(left, right)
    compiled = jax.jit(minkowski_sum)(left, right)

    assert_type(eager, ConstrainedZonotope)
    assert jnp.array_equal(eager.center, jnp.array([-1.0]))
    assert jnp.array_equal(eager.generator_matrix, jnp.array([[2.0, 3.0, 4.0]]))
    assert jnp.array_equal(
        eager.constraint_matrix,
        jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]]),
    )
    assert jnp.array_equal(eager.constraint_values, jnp.array([0.5, 0.0]))
    assert jnp.array_equal(compiled.center, eager.center)
    assert jnp.array_equal(compiled.constraint_matrix, eager.constraint_matrix)


def test_minkowski_sum_promotes_and_preserves_constraint_order() -> None:
    left = ConstrainedZonotope(
        jnp.array([1], dtype=jnp.float16),
        jnp.array([[2, 3]], dtype=jnp.float16),
        jnp.array([[4, 5], [6, 7]], dtype=jnp.float16),
        jnp.array([8, 9], dtype=jnp.float16),
    )
    right = ConstrainedZonotope(
        jnp.array([-2], dtype=jnp.float32),
        jnp.array([[10]], dtype=jnp.float32),
        jnp.array([[11]], dtype=jnp.float32),
        jnp.array([12], dtype=jnp.float32),
    )

    eager = minkowski_sum(left, right)
    compiled = jax.jit(minkowski_sum)(left, right)

    assert eager.dtype == jnp.float32
    assert jnp.array_equal(eager.center, jnp.array([-1.0]))
    assert jnp.array_equal(eager.generator_matrix, jnp.array([[2.0, 3.0, 10.0]]))
    assert jnp.array_equal(
        eager.constraint_matrix,
        jnp.array([[4.0, 5.0, 0.0], [6.0, 7.0, 0.0], [0.0, 0.0, 11.0]]),
    )
    assert jnp.array_equal(eager.constraint_values, jnp.array([8.0, 9.0, 12.0]))
    assert jnp.array_equal(compiled.center, eager.center)
    assert jnp.array_equal(compiled.generator_matrix, eager.generator_matrix)
    assert jnp.array_equal(compiled.constraint_matrix, eager.constraint_matrix)
    assert jnp.array_equal(compiled.constraint_values, eager.constraint_values)


def test_intersection_lifts_both_latent_vectors() -> None:
    left = ConstrainedZonotope([0], [[1]], jnp.empty((0, 1)), jnp.empty((0,)))
    right = ConstrainedZonotope([1], [[1]], jnp.empty((0, 1)), jnp.empty((0,)))

    eager = intersection(left, right)
    compiled = jax.jit(intersection)(left, right)

    assert_type(eager, ConstrainedZonotope)
    assert jnp.array_equal(eager.center, jnp.zeros(1))
    assert jnp.array_equal(eager.generator_matrix, jnp.array([[1.0, 0.0]]))
    assert jnp.array_equal(eager.constraint_matrix, jnp.array([[1.0, -1.0]]))
    assert jnp.array_equal(eager.constraint_values, jnp.ones(1))
    assert jnp.array_equal(compiled.generator_matrix, eager.generator_matrix)
    assert jnp.array_equal(compiled.constraint_matrix, eager.constraint_matrix)


def test_intersection_promotes_and_appends_coupling_constraints() -> None:
    left = ConstrainedZonotope(
        jnp.array([1, -1], dtype=jnp.float16),
        jnp.array([[2, 3], [4, 5]], dtype=jnp.float16),
        jnp.array([[6, 7]], dtype=jnp.float16),
        jnp.array([17], dtype=jnp.float16),
    )
    right = ConstrainedZonotope(
        jnp.array([9, 10], dtype=jnp.float32),
        jnp.array([[11], [12]], dtype=jnp.float32),
        jnp.array([[13], [14]], dtype=jnp.float32),
        jnp.array([18, 19], dtype=jnp.float32),
    )

    eager = intersection(left, right)
    compiled = jax.jit(intersection)(left, right)

    assert eager.dtype == jnp.float32
    assert jnp.array_equal(eager.center, jnp.array([1.0, -1.0]))
    assert jnp.array_equal(
        eager.generator_matrix,
        jnp.array([[2.0, 3.0, 0.0], [4.0, 5.0, 0.0]]),
    )
    assert jnp.array_equal(
        eager.constraint_matrix,
        jnp.array(
            [
                [6.0, 7.0, 0.0],
                [0.0, 0.0, 13.0],
                [0.0, 0.0, 14.0],
                [2.0, 3.0, -11.0],
                [4.0, 5.0, -12.0],
            ]
        ),
    )
    assert jnp.array_equal(
        eager.constraint_values,
        jnp.array([17.0, 18.0, 19.0, 8.0, 11.0]),
    )
    assert jnp.array_equal(compiled.center, eager.center)
    assert jnp.array_equal(compiled.generator_matrix, eager.generator_matrix)
    assert jnp.array_equal(compiled.constraint_matrix, eager.constraint_matrix)
    assert jnp.array_equal(compiled.constraint_values, eager.constraint_values)


def test_algebra_preserves_asymmetric_empty_shapes_under_vmap() -> None:
    left_generator_matrix = jnp.empty((1, 0), dtype=jnp.float16)
    left_constraint_matrix = jnp.empty((1, 0), dtype=jnp.float16)
    left_constraint_values = jnp.array([4], dtype=jnp.float16)
    right_generator_matrix = jnp.array([[2]], dtype=jnp.float32)
    right_constraint_matrix = jnp.empty((0, 1), dtype=jnp.float32)
    right_constraint_values = jnp.empty((0,), dtype=jnp.float32)
    left = ConstrainedZonotope(
        jnp.array([1], dtype=jnp.float16),
        left_generator_matrix,
        left_constraint_matrix,
        left_constraint_values,
    )
    right = ConstrainedZonotope(
        jnp.array([4], dtype=jnp.float32),
        right_generator_matrix,
        right_constraint_matrix,
        right_constraint_values,
    )

    eager_sum = minkowski_sum(left, right)
    eager_intersection = intersection(left, right)

    assert left.constraint_matrix.shape == (1, 0)
    assert right.constraint_matrix.shape == (0, 1)
    assert jnp.array_equal(eager_sum.center, jnp.array([5.0]))
    assert jnp.array_equal(eager_sum.generator_matrix, jnp.array([[2.0]]))
    assert jnp.array_equal(eager_sum.constraint_matrix, jnp.array([[0.0]]))
    assert jnp.array_equal(eager_sum.constraint_values, jnp.array([4.0]))
    assert jnp.array_equal(eager_intersection.center, jnp.array([1.0]))
    assert jnp.array_equal(eager_intersection.generator_matrix, jnp.array([[0.0]]))
    assert jnp.array_equal(
        eager_intersection.constraint_matrix, jnp.array([[0.0], [-2.0]])
    )
    assert jnp.array_equal(eager_intersection.constraint_values, jnp.array([4.0, 3.0]))

    left_centers = jnp.array([[1], [3]], dtype=jnp.float16)
    right_centers = jnp.array([[4], [8]], dtype=jnp.float32)
    batched_intersections = jax.jit(
        jax.vmap(
            lambda left_center, right_center: intersection(
                ConstrainedZonotope(
                    left_center,
                    left_generator_matrix,
                    left_constraint_matrix,
                    left_constraint_values,
                ),
                ConstrainedZonotope(
                    right_center,
                    right_generator_matrix,
                    right_constraint_matrix,
                    right_constraint_values,
                ),
            )
        )
    )(left_centers, right_centers)

    assert jnp.array_equal(
        batched_intersections.center, left_centers.astype(jnp.float32)
    )
    assert jnp.array_equal(batched_intersections.generator_matrix, jnp.zeros((2, 1, 1)))
    assert jnp.array_equal(
        batched_intersections.constraint_matrix,
        jnp.array([[[0.0], [-2.0]], [[0.0], [-2.0]]]),
    )
    assert jnp.array_equal(
        batched_intersections.constraint_values,
        jnp.array([[4.0, 3.0], [4.0, 5.0]]),
    )


def test_operations_vectorize_over_homogeneous_sets_and_differentiate() -> None:
    centers = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    generator_matrix = jnp.eye(2)
    constraint_matrix = jnp.array([[1.0, -1.0]])
    constraint_values = jnp.zeros(1)
    matrix = jnp.array([[2.0, 0.0], [1.0, -1.0]])

    batched_sets = jax.vmap(
        lambda center: ConstrainedZonotope(
            center,
            generator_matrix,
            constraint_matrix,
            constraint_values,
        )
    )(centers)
    images = jax.jit(jax.vmap(lambda convex_set: convex_set.affine_map(matrix)))(
        batched_sets
    )

    def center_sum(center: jax.Array) -> jax.Array:
        convex_set = ConstrainedZonotope(
            center,
            generator_matrix,
            constraint_matrix,
            constraint_values,
        )
        return jnp.sum(convex_set.affine_map(matrix).center)

    gradient = jax.jit(jax.grad(center_sum))(jnp.zeros(2))

    assert jnp.array_equal(images.center, centers @ matrix.T)
    assert jnp.array_equal(gradient, jnp.sum(matrix, axis=0))


def test_algebra_rejects_dimension_mismatches_and_mixed_representations() -> None:
    one_dimensional = ConstrainedZonotope(
        [0], [[1]], jnp.empty((0, 1)), jnp.empty((0,))
    )
    two_dimensional = ConstrainedZonotope(
        [0, 0], jnp.eye(2), jnp.empty((0, 2)), jnp.empty((0,))
    )
    zonotope = Zonotope([0], [[1]])

    with pytest.raises(ValueError, match="dimensions must match"):
        minkowski_sum(one_dimensional, two_dimensional)
    with pytest.raises(ValueError, match="dimensions must match"):
        intersection(one_dimensional, two_dimensional)
    with pytest.raises(TypeError, match="not implemented"):
        cast(Any, minkowski_sum)(one_dimensional, zonotope)
    with pytest.raises(TypeError, match="not implemented"):
        cast(Any, intersection)(one_dimensional, zonotope)
