from __future__ import annotations

from typing import Self

from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, ScalarLike

from .abstract_set import AbstractSet


class Ellipsoid(AbstractSet):
    """
    Ellipsoid in quadratic form:

        E = { x : (x - c)^T Q^{-1} (x - c) <= 1 }

    where Q is intended to be symmetric positive semidefinite (PSD).
    """

    c: Float[Array, " n"]
    Q: Float[Array, " n n"]

    _psd_tol: ScalarLike = jnp.sqrt(jnp.finfo(float).eps)
    _pinv_rcond: ScalarLike = jnp.sqrt(jnp.finfo(float).eps)

    def __init__(self, c: Float[ArrayLike, " n"], Q: Float[ArrayLike, " n n"]):
        c_arr = jnp.asarray(c)
        Q_arr = jnp.asarray(Q)

        if c_arr.ndim != 1:
            raise ValueError(f"c must be 1D, got shape {c_arr.shape}")
        if Q_arr.ndim != 2:
            raise ValueError(f"Q must be 2D, got shape {Q_arr.shape}")
        if Q_arr.shape[0] != Q_arr.shape[1]:
            raise ValueError(f"Q must be square, got shape {Q_arr.shape}")
        if Q_arr.shape[0] != c_arr.shape[0]:
            raise ValueError(
                f"Q shape {Q_arr.shape} incompatible with c shape {c_arr.shape}"
            )

        self.c = c_arr
        self.Q = 0.5 * (Q_arr + Q_arr.T)

    @property
    def dim(self) -> int:
        return int(self.c.shape[0])

    @classmethod
    def quadratic_form(
        cls, c: Float[ArrayLike, " n"], Q: Float[ArrayLike, " n n"]
    ) -> Ellipsoid:
        return cls(c, Q)

    @classmethod
    def full_rank(
        cls, c: Float[ArrayLike, " n"], G: Float[ArrayLike, " n n"]
    ) -> Ellipsoid:
        G_arr = jnp.asarray(G)
        return cls(c, G_arr @ G_arr.T)

    @classmethod
    def center_radius(
        cls, c: Float[ArrayLike, " n"], r: Float[ArrayLike, ""]
    ) -> Ellipsoid:
        c_arr = jnp.asarray(c)
        n = c_arr.shape[0]
        r_arr = jnp.asarray(r)
        return cls(c_arr, (r_arr**2) * jnp.eye(n, dtype=c_arr.dtype))

    @property
    def is_empty(self) -> Bool[Array, ""]:
        finite_ok = jnp.all(jnp.isfinite(self.c)) & jnp.all(jnp.isfinite(self.Q))
        eigvals = jnp.linalg.eigvalsh(self.Q)
        psd_ok = jnp.all(eigvals >= -jnp.asarray(self._psd_tol))
        return jnp.asarray(~(finite_ok & psd_ok))

    @property
    def is_bounded(self) -> Bool[Array, ""]:
        return jnp.asarray(~self.is_empty)

    def _check_vector_shape(self, x: Array, name: str) -> None:
        if x.ndim != 1 or x.shape[0] != self.dim:
            raise ValueError(f"{name} must have shape ({self.dim},), got {x.shape}")

    def contains(self, x: Float[ArrayLike, " n"]) -> Bool[Array, ""]:
        x_arr = jnp.asarray(x)
        self._check_vector_shape(x_arr, "x")

        rel = x_arr - self.c

        pinv_Q = jnp.linalg.pinv(self.Q, rcond=self._pinv_rcond)
        quad = rel @ (pinv_Q @ rel)

        projected = self.Q @ (pinv_Q @ rel)
        residual = rel - projected
        residual_ok = jnp.linalg.norm(residual) <= (10.0 * self._psd_tol)

        return jnp.asarray(
            (~self.is_empty) & residual_ok & (quad <= (1.0 + 10.0 * self._psd_tol))
        )

    def support_point(self, direction: Float[ArrayLike, " n"]) -> Float[Array, " n"]:
        d = jnp.asarray(direction)
        self._check_vector_shape(d, "direction")

        if jnp.all(d == 0):
            return self.c

        if bool(self.is_empty):
            raise ValueError("Support point is undefined for an empty ellipsoid.")

        Qd = self.Q @ d
        denom_sq = d @ Qd
        denom = jnp.sqrt(jnp.maximum(denom_sq, 0.0))

        step = jnp.where(denom > 0, Qd / denom, jnp.zeros_like(Qd))
        return self.c + step

    def closest_point(
        self, x: Float[ArrayLike, " n"], p: ScalarLike = 2
    ) -> Float[Array, " n"]:
        if p != 2:
            raise ValueError("Ellipsoid.closest_point currently supports only p=2.")

        x_arr = jnp.asarray(x)
        self._check_vector_shape(x_arr, "x")

        if bool(self.is_empty):
            raise ValueError("Projection is undefined for an empty ellipsoid.")

        rel = x_arr - self.c

        eigvals, eigvecs = jnp.linalg.eigh(self.Q)
        eigvals = jnp.maximum(eigvals, 0.0)
        z = eigvecs.T @ rel

        active = eigvals > self._psd_tol
        null_violation = jnp.any(
            jnp.where(active, False, jnp.abs(z) > (10.0 * self._psd_tol))
        )

        quad = jnp.sum(jnp.where(active, (z * z) / eigvals, 0.0))
        inside = (~null_violation) & (quad <= (1.0 + 10.0 * self._psd_tol))

        def phi(lam: Array) -> Array:
            denom = eigvals + lam
            terms = jnp.where(active, (eigvals * (z * z)) / (denom * denom), 0.0)
            return jnp.sum(terms)

        def project_outside(_: Array) -> Array:
            lam_lo = jnp.asarray(0.0, dtype=self.Q.dtype)
            lam_hi = jnp.asarray(1.0, dtype=self.Q.dtype)

            def cond_expand(state):
                lam_hi_local, k = state
                return (phi(lam_hi_local) > 1.0) & (k < 60)

            def body_expand(state):
                lam_hi_local, k = state
                return (lam_hi_local * 2.0, k + 1)

            lam_hi, _ = lax.while_loop(
                cond_expand, body_expand, (lam_hi, jnp.asarray(0))
            )

            def body_bisect(state, _):
                lo, hi = state
                mid = 0.5 * (lo + hi)
                go_left = phi(mid) > 1.0
                lo = jnp.where(go_left, mid, lo)
                hi = jnp.where(go_left, hi, mid)
                return (lo, hi), None

            (lam_lo, lam_hi), _ = lax.scan(body_bisect, (lam_lo, lam_hi), length=80)
            lam = 0.5 * (lam_lo + lam_hi)

            scale = jnp.where(active, eigvals / (eigvals + lam), 0.0)
            w = scale * z
            return self.c + eigvecs @ w

        return lax.cond(
            inside, lambda _: x_arr, project_outside, operand=jnp.asarray(0.0)
        )

    def linear_map(self, M: Float[ArrayLike, " m n"] | Float[ArrayLike, ""]) -> Self:
        M_arr = jnp.asarray(M)

        if bool(self.is_empty):
            return type(self)(self.c, self.Q)

        if M_arr.ndim == 0:
            s = M_arr
            return type(self)(s * self.c, (s * s) * self.Q)

        if M_arr.ndim != 2:
            raise ValueError(
                f"M must be a scalar or 2D matrix, got shape {M_arr.shape}"
            )
        if M_arr.shape[1] != self.dim:
            raise ValueError(f"M must have shape (m, {self.dim}), got {M_arr.shape}")

        c_new = M_arr @ self.c
        Q_new = M_arr @ self.Q @ M_arr.T
        return type(self)(c_new, Q_new)

    def translate(self, b: Float[ArrayLike, " n"]) -> Self:
        b_arr = jnp.asarray(b)
        self._check_vector_shape(b_arr, "b")
        if bool(self.is_empty):
            return type(self)(self.c, self.Q)
        return type(self)(self.c + b_arr, self.Q)
