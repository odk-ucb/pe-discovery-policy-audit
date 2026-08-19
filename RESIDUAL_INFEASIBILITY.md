# Residual infeasible random policies (TrpB4)

`RAND:draw_13` and `RAND:draw_16` fail with:
`generator produced 4 < batch 24 at round 3`

Cause: `local_hamming_ball` + small radius late in the campaign exhausts the unevaluated neighborhood.
This is **true design-space infeasibility**, not a missing cache. ITT analysis already treats such policies as anchor-level.

Not "fixed" by silently expanding the pool (that would change the policy).
