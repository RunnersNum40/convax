from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float, Real, ScalarLike

from convax._utils import (
    as_float_array,
    require_finite,
    require_scalar,
    require_vector,
)


@final
class SupportResult(eqx.Module):
    """A support-function value and corresponding support point.

    Args:
        value: Scalar support value.
        point: Support point with shape ``(ambient_dimension,)``.

    Attributes:
        value: Scalar support value.
        point: Support point with shape ``(ambient_dimension,)``.

    Raises:
        TypeError: If either input contains complex values.
        ValueError: If ``value`` is not scalar or ``point`` is not a vector.
        EquinoxRuntimeError: If ``value`` or ``point`` is nonfinite.
    """

    value: Float[Array, ""]
    point: Float[Array, "ambient_dimension"]

    def __init__(
        self,
        value: ScalarLike,
        point: Real[ArrayLike, "ambient_dimension"] | Sequence[float | int],
    ) -> None:
        value = as_float_array(value)
        point = as_float_array(point)
        require_scalar("value", value)
        require_vector("point", point)
        dtype = jnp.result_type(value.dtype, point.dtype)
        self.value = require_finite("value", value.astype(dtype))
        self.point = require_finite("point", point.astype(dtype))


@final
class AxisAlignedBounds(eqx.Module):
    """Tight coordinate-wise bounds derived from support values.

    Args:
        lower: Lower bounds with shape ``(ambient_dimension,)``.
        upper: Upper bounds with shape ``(ambient_dimension,)``.

    Attributes:
        lower: Lower bounds with shape ``(ambient_dimension,)``.
        upper: Upper bounds with shape ``(ambient_dimension,)``.

    Raises:
        TypeError: If either input contains complex values.
        ValueError: If an input is not a vector or their shapes differ.
        EquinoxRuntimeError: If ``lower`` or ``upper`` is nonfinite, or the
            bounds are reversed.
    """

    lower: Float[Array, "ambient_dimension"]
    upper: Float[Array, "ambient_dimension"]

    def __init__(
        self,
        lower: Real[ArrayLike, "ambient_dimension"] | Sequence[float | int],
        upper: Real[ArrayLike, "ambient_dimension"] | Sequence[float | int],
    ) -> None:
        lower = as_float_array(lower)
        upper = as_float_array(upper)
        require_vector("lower", lower)
        require_vector("upper", upper)
        if lower.shape != upper.shape:
            raise ValueError(
                "lower and upper bounds must have matching shapes, got "
                f"{lower.shape} and {upper.shape}"
            )
        dtype = jnp.result_type(lower.dtype, upper.dtype)
        lower = require_finite("lower", lower.astype(dtype))
        upper = require_finite("upper", upper.astype(dtype))
        lower, upper = eqx.error_if(
            (lower, upper),
            jnp.any(lower > upper),
            "lower bounds must not exceed upper bounds",
        )
        self.lower = lower
        self.upper = upper
