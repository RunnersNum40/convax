import jax
import jax.numpy as jnp
import pytest

from convax import VertexPolytope


def test_support_selects_maximizing_vertex() -> None:
    polytope = VertexPolytope([[0, 0], [2, 0], [0, 1]])

    support = polytope.support(jnp.array([1.0, 0.5]))

    assert jnp.allclose(support.value, 2.0)
    assert jnp.allclose(support.point, jnp.array([2.0, 0.0]))


def test_support_vectorizes_over_directions() -> None:
    polytope = VertexPolytope([[0, 0], [2, 0], [0, 1]])

    values = jax.jit(jax.vmap(polytope.support_value))(jnp.eye(2))

    assert jnp.allclose(values, jnp.array([2.0, 1.0]))


def test_constructor_rejects_empty_vertices() -> None:
    with pytest.raises(ValueError, match="at least one"):
        VertexPolytope(jnp.empty((0, 2)))
