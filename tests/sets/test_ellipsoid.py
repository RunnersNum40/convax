import jax
import jax.numpy as jnp
import pytest

from convax import Ellipsoid


def test_support_matches_closed_form() -> None:
    ellipsoid = Ellipsoid([1, -1], [[2, 0], [0, 1]])

    support = ellipsoid.support(jnp.array([1.0, 0.0]))

    assert jnp.allclose(support.value, 3.0)
    assert jnp.allclose(support.point, jnp.array([3.0, -1.0]))


def test_zero_direction_returns_center() -> None:
    ellipsoid = Ellipsoid([1, -1], [[2, 0], [0, 1]])

    support = ellipsoid.support(jnp.zeros(2))

    assert jnp.allclose(support.value, 0.0)
    assert jnp.allclose(support.point, ellipsoid.center)


def test_rank_deficient_containment() -> None:
    ellipsoid = Ellipsoid([0, 0], [[1], [0]])

    assert ellipsoid.contains(jnp.array([0.5, 0.0]))
    assert not ellipsoid.contains(jnp.array([2.0, 0.0]))
    assert not ellipsoid.contains(jnp.array([0.5, 0.1]))


def test_singleton_containment() -> None:
    singleton = Ellipsoid(jnp.array([1.0, 2.0]), jnp.empty((2, 0)))

    assert singleton.contains(jnp.array([1.0, 2.0]))
    assert not singleton.contains(jnp.array([1.0, 2.1]))


def test_support_is_jittable_vectorizable_and_differentiable() -> None:
    center = jnp.array([1.0, -1.0])
    generator_matrix = jnp.array([[2.0, 0.0], [0.0, 1.0]])
    ellipsoid = Ellipsoid(center, generator_matrix)
    directions = jnp.eye(2)

    eager_values = jax.vmap(ellipsoid.support_value)(directions)
    compiled_values = jax.jit(jax.vmap(ellipsoid.support_value))(directions)
    center_gradient = jax.grad(
        lambda candidate_center: Ellipsoid(
            candidate_center, generator_matrix
        ).support_value(jnp.array([1.0, 2.0]))
    )(center)

    assert jnp.allclose(compiled_values, eager_values)
    assert jnp.allclose(center_gradient, jnp.array([1.0, 2.0]))


def test_constructor_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="rows must match"):
        Ellipsoid(jnp.zeros(2), jnp.zeros((3, 1)))

    with pytest.raises(ValueError, match="must be a vector"):
        Ellipsoid(jnp.zeros((1, 2)), jnp.zeros((2, 1)))
