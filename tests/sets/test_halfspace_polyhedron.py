import jax
import jax.numpy as jnp
import pytest

from convax import HalfspacePolyhedron


def test_containment_checks_inequalities_and_equalities() -> None:
    polyhedron = HalfspacePolyhedron(
        [[1, 0], [0, 1]],
        [1, 1],
        [[1, -1]],
        [0],
    )

    assert polyhedron.contains(jnp.array([0.5, 0.5]))
    assert not polyhedron.contains(jnp.array([2.0, 2.0]))
    assert not polyhedron.contains(jnp.array([0.5, 0.4]))


def test_containment_is_jittable_and_vectorizable() -> None:
    box = HalfspacePolyhedron(
        [[1, 0], [-1, 0], [0, 1], [0, -1]],
        [1, 1, 1, 1],
    )
    points = jnp.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.0]])

    contained = jax.jit(jax.vmap(box.contains))(points)

    assert jnp.array_equal(contained, jnp.array([True, True, False]))


def test_empty_equality_arrays_are_materialized() -> None:
    polyhedron = HalfspacePolyhedron(jnp.empty((0, 2)), jnp.empty((0,)))

    assert polyhedron.equality_matrix.shape == (0, 2)
    assert polyhedron.equality_values.shape == (0,)
    assert polyhedron.contains(jnp.array([100.0, -100.0]))


def test_constructor_rejects_incomplete_equalities() -> None:
    with pytest.raises(ValueError, match="provided together"):
        HalfspacePolyhedron(jnp.eye(2), jnp.ones(2), jnp.eye(2))

    with pytest.raises(ValueError, match="provided together"):
        HalfspacePolyhedron(jnp.eye(2), jnp.ones(2), equality_values=jnp.ones(2))


def test_constructor_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="rows must match"):
        HalfspacePolyhedron(jnp.eye(2), jnp.ones(3))
