from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import ScalarLike, TypeCheckError

from convax._utils import VectorLike
from convax.sets import AxisAlignedBounds, SupportResult


def test_support_result_normalizes_array_like_inputs() -> None:
    result = SupportResult(np.array(2.0, dtype=np.float16), [1, 2])

    assert isinstance(result.value, jax.Array)
    assert isinstance(result.point, jax.Array)
    assert result.value.shape == ()
    assert result.point.shape == (2,)
    expected_dtype = jnp.result_type(jnp.float16, jnp.result_type(0.0))
    assert result.value.dtype == expected_dtype
    assert result.point.dtype == expected_dtype


def test_axis_aligned_bounds_normalize_array_like_inputs() -> None:
    bounds = AxisAlignedBounds(
        np.array([-1, -2], dtype=np.float16),
        np.array([3, 4], dtype=np.float32),
    )

    assert isinstance(bounds.lower, jax.Array)
    assert isinstance(bounds.upper, jax.Array)
    expected_dtype = jnp.result_type(jnp.float16, jnp.float32)
    assert bounds.lower.dtype == expected_dtype
    assert bounds.upper.dtype == expected_dtype
    assert jnp.array_equal(bounds.lower, jnp.array([-1.0, -2.0]))
    assert jnp.array_equal(bounds.upper, jnp.array([3.0, 4.0]))


def test_result_construction_is_jittable() -> None:
    support = jax.jit(SupportResult)(jnp.array(2.0), jnp.array([1.0, 2.0]))
    bounds = jax.jit(AxisAlignedBounds)(jnp.array([-1.0, -2.0]), jnp.array([3.0, 4.0]))

    assert jnp.allclose(support.value, 2.0)
    assert jnp.allclose(support.point, jnp.array([1.0, 2.0]))
    assert jnp.allclose(bounds.lower, jnp.array([-1.0, -2.0]))
    assert jnp.allclose(bounds.upper, jnp.array([3.0, 4.0]))


def test_invalid_result_shape_fails_during_jit_tracing() -> None:
    with pytest.raises(TypeCheckError, match="parameter 'value'"):
        jax.jit(SupportResult)(jnp.ones(1), jnp.ones(2))


def test_support_result_rejects_invalid_shapes() -> None:
    with pytest.raises(TypeCheckError, match="parameter 'value'"):
        SupportResult(cast(ScalarLike, [1]), [1, 2])

    with pytest.raises(TypeCheckError, match="parameter 'point'"):
        SupportResult(1, cast(VectorLike, [[1, 2]]))


def test_axis_aligned_bounds_reject_invalid_shapes() -> None:
    with pytest.raises(TypeCheckError, match="parameter 'lower'"):
        AxisAlignedBounds(cast(VectorLike, [[-1, -2]]), [3, 4])

    with pytest.raises(ValueError, match="matching shapes"):
        AxisAlignedBounds([-1], [2, 3])


def test_result_constructors_reject_complex_inputs() -> None:
    with pytest.raises(TypeError, match="requires real-valued arrays"):
        SupportResult(jnp.array(1.0 + 1.0j), [1])

    with pytest.raises(TypeCheckError, match="parameter 'lower'"):
        AxisAlignedBounds(jnp.array([-1.0 + 1.0j]), [1])
