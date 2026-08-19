# Robust Affine Constraints

Suppose an output must satisfy \(A_y y \leq b_y\) with

\[
y = Cx + d + w, \qquad w \in W.
\]

For each row \(a_i^\mathsf{T}\) of \(A_y\), the largest disturbance contribution is \(h_W(a_i)\), so the output constraint holds for every disturbance exactly when

\[
A_y Cx \leq b_y - A_y d -
\begin{bmatrix}
h_W(a_1) & \cdots & h_W(a_m)
\end{bmatrix}^\mathsf{T}.
\]

The example computes all margins with `jax.vmap`, tightens output limits, and uses an affine preimage to express them in state coordinates:

The robust preimage is intersected with independent state limits, and a vectorized compiled query classifies candidate states. This construction handles inequality limits; a nontrivial additive disturbance generally cannot preserve an output equality for every realization, so the example rejects equality constraints.

<!-- fmt:off -->
```python title="examples/robust_constraints.py"
--8<-- "examples/robust_constraints.py"
```
<!-- fmt:on -->
