# Input-Dependent Bilinear Mixing: Exploratory Architecture Hypothesis

## Status and separation from the baseline

This is a research hypothesis, not an implemented model or an experimental
result. It is a follow-up track separate from the TinyStories byte-level
Transformer baseline. The baseline remains unchanged: fixed random byte LUT,
full causal attention, TeLU FFN, next-byte cross entropy, and AdamW.

## Proposed operator form

For a hidden state matrix \(X \in \mathbb{R}^{T \times d}\), consider

\[
X_{l+1}
= X_l + \Phi_l\left(
\sum_{r=1}^{R_l} A_{l,r}(X_l) X_l B_{l,r}(X_l)
\right).
\]

One possible explicit branch is

\[
Z_r = A_r(X) X B_r(X),\qquad
F(X) = \sum_{r=1}^{R} C_r\,\sigma(Z_r+b_r)\,D_r.
\]

For this notation, a shape-preserving dense interpretation would use
\(A_r(X)\in\mathbb{R}^{T\times T}\) and
\(B_r(X)\in\mathbb{R}^{d\times d}\). This defines the mathematical
possibility, not a practical implementation: materializing input-dependent
dense matrices could be too expensive. Low-rank factors, structured matrices,
or matrix-free operators need separate study. The dimensions and role of
\(\Phi_l\), \(C_r\), and \(D_r\) are also open design choices.

With column-major vectorization,
\(\operatorname{vec}(AXB)=(B^\top\otimes A)\operatorname{vec}(X)\).
Thus each fixed two-sided branch is a Kronecker-structured linear mixing
before its nonlinearity. When \(A\) and \(B\) depend on \(X\), the operator
also selects its mixing directions from the current input.

## Core questions

1. **Direction generation:** Which useful sequence-space and channel-space
   directions can \(A_r(X)\) and \(B_r(X)\) represent? Can low-rank or
   factorized generators discover them without constructing dense matrices?
2. **Nonlinear rank:** How does approximation error change with branch count
   \(R\), factor rank, and the dimensions of each branch? What is the smallest
   \(R\) that achieves a target error?
3. **Input dependence:** When do input-generated directions outperform fixed
   directions? What parameterization is expressive enough without unstable
   feedback or excessive compute?
4. **Dynamic complexity:** Can \(R(X)\) vary by input? A useful design must
   account for routing/selection cost, predictable compute budgets, and
   stability; adding branches alone is not evidence that a learned support was
   discovered.
5. **Identifiability and regularity:** What normalization or constraints avoid
   equivalent rescalings among \(A,B,C,D\)? How should support complexity or
   variation be measured?

## Relation to existing modules

- **Attention** performs input-dependent mixing across sequence positions,
  while its values and output projection mix channels.
- **FFN** applies a nonlinearity in a learned channel basis independently at
  each position.
- **Bilinear mixing** may couple sequence and channel structure around the
  nonlinearity. It does not automatically subsume attention or FFN, and it is
  not assumed to be more expressive or efficient without measurement.

The scalar bilinear projection \(u^\top Xv\) is a useful minimal probe. It
selects one joint sequence/channel direction, analogous to a ridge feature;
many such features can then be combined. Active-subspace and
variation/Barron-space viewpoints may help formulate low-complexity cases,
but universal-approximation results alone do not establish learnability,
sample efficiency, or a small required \(R\).

## Suggested evaluation sequence

1. Define synthetic matrix-valued target operators with known separable,
   low-rank, and non-separable interactions.
2. Measure approximation error against \(R\), factor rank, parameter count,
   and FLOPs; include a conventional MLP/attention reference where applicable.
3. Compare fixed directions with input-generated directions while holding
   parameter and compute budgets as close as practical.
4. Only then test dynamic \(R(X)\), reporting both quality and per-input
   compute, including the routing cost.
5. If the operator passes these probes, create a separately configured
   language-model experiment. Do not mix it into the baseline or change
   activation, LUT initialization, position encoding, and optimizer at the
   same time.

## Reading trail

- Van Loan & Pitsianis, [Approximation with Kronecker Products](https://ecommons.cornell.edu/entities/publication/ad6a5c71-2632-4652-b33a-72016d7abc4d)
- Lu et al., [Learning nonlinear operators via DeepONet](https://doi.org/10.1038/s42256-021-00302-5)
- [Error estimates for DeepONets](https://doi.org/10.1093/imatrm/tnac001)
- Next: Tensor Networks, Koopman theory, ridge-function approximation, and
  active subspaces. Exact papers for the latter topics remain to be selected
  and verified before adding formal citations.
