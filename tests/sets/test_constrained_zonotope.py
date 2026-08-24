import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from convax import ConstrainedZonotope


def test_constructor_normalizes_arrays_and_promotes_common_dtype() -> None:
    constrained_zonotope = ConstrainedZonotope(
        jnp.array([1, 2], dtype=jnp.float16),
        jnp.array([[1], [2]], dtype=jnp.float16),
        jnp.array([[1]], dtype=jnp.float32),
        jnp.array([0], dtype=jnp.float16),
    )

    assert constrained_zonotope.ambient_dimension == 2
    assert constrained_zonotope.dtype == jnp.float32
    assert constrained_zonotope.center.dtype == jnp.float32
    assert constrained_zonotope.generator_matrix.dtype == jnp.float32
    assert constrained_zonotope.constraint_matrix.dtype == jnp.float32
    assert constrained_zonotope.constraint_values.dtype == jnp.float32
    assert isinstance(constrained_zonotope.center, jax.Array)


def test_constructor_accepts_explicit_empty_constraints_and_generators() -> None:
    constrained_zonotope = ConstrainedZonotope(
        [1, 2],
        jnp.empty((2, 0)),
        jnp.empty((0, 0)),
        jnp.empty((0,)),
    )

    assert constrained_zonotope.generator_matrix.shape == (2, 0)
    assert constrained_zonotope.constraint_matrix.shape == (0, 0)
    assert constrained_zonotope.constraint_values.shape == (0,)


def test_constructor_accepts_infeasible_and_redundant_constraints() -> None:
    constrained_zonotope = ConstrainedZonotope(
        [0],
        [[1]],
        [[0], [0]],
        [1, 1],
    )

    assert constrained_zonotope.constraint_matrix.shape == (2, 1)
    assert jnp.array_equal(constrained_zonotope.constraint_values, jnp.ones(2))


@pytest.mark.parametrize(
    (
        "center",
        "generator_matrix",
        "constraint_matrix",
        "constraint_values",
        "parameter",
    ),
    [
        (
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            "center",
        ),
        (
            jnp.zeros(1),
            jnp.zeros(1),
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            "generator_matrix",
        ),
        (
            jnp.zeros(2),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            "generator_matrix",
        ),
        (
            jnp.zeros(1),
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            jnp.zeros(1),
            "constraint_matrix",
        ),
        (
            jnp.zeros(1),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            "constraint_values",
        ),
        (
            jnp.zeros(1),
            jnp.zeros((1, 2)),
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            "constraint_matrix",
        ),
        (
            jnp.zeros(1),
            jnp.zeros((1, 1)),
            jnp.zeros((2, 1)),
            jnp.zeros(1),
            "constraint_values",
        ),
    ],
)
def test_constructor_rejects_invalid_shapes(
    center: jax.Array,
    generator_matrix: jax.Array,
    constraint_matrix: jax.Array,
    constraint_values: jax.Array,
    parameter: str,
) -> None:
    with pytest.raises(TypeCheckError, match=f"parameter '{parameter}'"):
        ConstrainedZonotope(
            center,
            generator_matrix,
            constraint_matrix,
            constraint_values,
        )


def test_constructor_rejects_complex_fields() -> None:
    real_vector = jnp.ones(1)
    real_matrix = jnp.ones((1, 1))
    complex_vector = jnp.ones(1, dtype=jnp.complex64)
    complex_matrix = jnp.ones((1, 1), dtype=jnp.complex64)

    with pytest.raises(TypeCheckError, match="parameter 'center'"):
        ConstrainedZonotope(complex_vector, real_matrix, real_matrix, real_vector)
    with pytest.raises(TypeCheckError, match="parameter 'generator_matrix'"):
        ConstrainedZonotope(real_vector, complex_matrix, real_matrix, real_vector)
    with pytest.raises(TypeCheckError, match="parameter 'constraint_matrix'"):
        ConstrainedZonotope(real_vector, real_matrix, complex_matrix, real_vector)
    with pytest.raises(TypeCheckError, match="parameter 'constraint_values'"):
        ConstrainedZonotope(real_vector, real_matrix, real_matrix, complex_vector)


def test_constructor_is_jittable_and_vectorizable() -> None:
    centers = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    generator_matrix = jnp.eye(2)
    constraint_matrix = jnp.array([[1.0, -1.0]])
    constraint_values = jnp.zeros(1)

    batched_sets = jax.jit(
        jax.vmap(
            lambda center: ConstrainedZonotope(
                center,
                generator_matrix,
                constraint_matrix,
                constraint_values,
            )
        )
    )(centers)

    assert jnp.array_equal(batched_sets.center, centers)
    assert batched_sets.generator_matrix.shape == (2, 2, 2)
    assert batched_sets.constraint_matrix.shape == (2, 1, 2)
