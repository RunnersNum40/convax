# Reachable Set Envelopes

For one step of uncertain linear dynamics,

\[
x^+ = Ax + Bu + w,
\]

with initial state set \(X\), control set \(U\), and additive process-noise set \(W\), the reachable set is

\[
AX \oplus BU \oplus W,
\]

where \(\oplus\) denotes a Minkowski sum.

The example compares nominal and emergency controls by constructing their convex
envelope

\[
\operatorname{conv}(AX \oplus BU_\mathrm{nominal},
                    AX \oplus BU_\mathrm{emergency}) \oplus W.
\]

The construction uses the free operations `affine_map`, `minkowski_sum`, and
`convex_hull`, so each step selects its strongest exact set type.

<!-- fmt:off -->
```python title="examples/reachable_sets.py"
--8<-- "examples/reachable_sets.py"
```
<!-- fmt:on -->
