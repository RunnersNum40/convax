# Differentiable Geometry

Place a local ellipsoidal footprint \(X_0\) at planar pose \(p = (t_x, t_y, \theta)\):

\[
X(p) = R(\theta)X_0 + t.
\]

For an obstacle boundary \(n^\mathsf{T}x \leq \beta\), define signed clearance from the worst footprint point as

\[
c(p) = \beta - h_{X(p)}(n).
\]

!!! note

    Support functions can be nonsmooth at maximizing-point ties or active-set changes.

Positive clearance means the full footprint satisfies the halfspace. The
full-rank ellipsoid and nonzero obstacle normal make this query smooth at the
shown pose, while `affine_map` retains the ellipsoid type.

<!-- fmt:off -->
```python title="examples/differentiable_geometry.py"
--8<-- "examples/differentiable_geometry.py"
```
<!-- fmt:on -->
