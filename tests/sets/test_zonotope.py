import jax
import jax.numpy as jnp
import pytest

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


def test_constructor_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="rows must match"):
        Zonotope(jnp.zeros(2), jnp.zeros((3, 1)))
