import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from convax import Zonotope


def test_support_matches_generator_formula() -> None:
    zonotope = Zonotope([1, -1], [[2, 0], [0, 1]])

    support = zonotope.support(jnp.array([1.0, -2.0]))

    assert jnp.allclose(support.value, 7.0)
    assert jnp.allclose(support.point, jnp.array([3.0, -2.0]))


def test_zero_generator_coefficients_select_center_coefficients() -> None:
    zonotope = Zonotope([1, 2], [[1, 0], [0, 1]])

    support = zonotope.support(jnp.zeros(2))

    assert jnp.allclose(support.point, zonotope.center)
    assert jnp.allclose(support.value, 0.0)


def test_axis_aligned_bounds_are_tight_and_jittable() -> None:
    zonotope = Zonotope([1, -1], [[2, 0], [0, 1]])

    eager_bounds = zonotope.axis_aligned_bounds()
    compiled_bounds = jax.jit(lambda convex_set: convex_set.axis_aligned_bounds())(
        zonotope
    )

    assert jnp.allclose(eager_bounds.lower, jnp.array([-1.0, -2.0]))
    assert jnp.allclose(eager_bounds.upper, jnp.array([3.0, 0.0]))
    assert jnp.allclose(compiled_bounds.lower, eager_bounds.lower)
    assert jnp.allclose(compiled_bounds.upper, eager_bounds.upper)


def test_affine_map_preserves_zonotope_representation() -> None:
    zonotope = Zonotope([1, -1], [[2, 0], [0, 1]])
    matrix = jnp.array([[1.0, 2.0]])
    offset = jnp.array([0.5])

    image = zonotope.affine_map(matrix, offset)

    assert isinstance(image, Zonotope)
    assert jnp.array_equal(image.center, matrix @ zonotope.center + offset)
    assert jnp.array_equal(image.generator_matrix, matrix @ zonotope.generator_matrix)


def test_affine_map_promotes_mixed_dtypes() -> None:
    center = jnp.array([1.0, -2.0], dtype=jnp.float16)
    generator_matrix = jnp.array([[2.0, 0.0], [0.0, 1.0]], dtype=jnp.float16)
    matrix = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    offset = jnp.array([0.5], dtype=jnp.float16)
    zonotope = Zonotope(center, generator_matrix)
    expected_center = matrix @ center.astype(jnp.float32) + offset.astype(jnp.float32)
    expected_generator_matrix = matrix @ generator_matrix.astype(jnp.float32)

    eager_image = zonotope.affine_map(matrix, offset)
    compiled_image = jax.jit(lambda convex_set: convex_set.affine_map(matrix, offset))(
        zonotope
    )

    for image in (eager_image, compiled_image):
        assert image.center.dtype == jnp.float32
        assert image.generator_matrix.dtype == jnp.float32
        assert jnp.array_equal(image.center, expected_center)
        assert jnp.array_equal(image.generator_matrix, expected_generator_matrix)


def test_constructor_rejects_incompatible_shapes() -> None:
    with pytest.raises(TypeCheckError, match="parameter 'generator_matrix'"):
        Zonotope(jnp.zeros(2), jnp.zeros((3, 1)))
