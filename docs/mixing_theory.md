# Why partial-sum auxiliaries mix slowly: a linear-Gaussian theory

Code: [`tomo/mixing_theory.py`](../tomo/mixing_theory.py), tests: [`tests/test_mixing_theory.py`](../tests/test_mixing_theory.py),
experiments: [`experiments/mixing_theory.py`](../experiments/mixing_theory.py) (raw numbers in `results/mixing_theory/*.json`).

The README reports that dense Gibbs mixes in about 2 sweeps while I-chain and I-tree need hundreds. This note turns that
observation into a prediction. Everything below is exact linear algebra on a continuous relaxation, checked against
(a) a simulation of the linear Gibbs chain and (b) the actual discrete Ising samplers.

## 1. Setting

**Continuous relaxation.** Keep the energies of `ebm_chain` / `ebm_tree` but let pixels $x$ and auxiliary partial sums $z$ be
real. The joint law is Gaussian, $N(\mu, P^{-1})$ with sparse precision $P$ over $(x,z)$:

```math
E = \sum_i\Big[\sum_{k}\frac{(z_{ik}-z_{i,k-1}-t_{ik}x_{j_k})^2}{2\tau_{ik}^2} + \frac{(b_i-z_{i,n})^2}{2 s_i^2}\Big] + \frac{\lambda}{2}x^\top L x ,
\qquad s_i^2=\sigma_i^2-\sum_k\tau_{ik}^2 \ \text{(compensated)}.
```

`build_joint` assembles $P$ for `dense` ($P=T^\top W T+\lambda L$), `chain` and `tree`, with the same $\lambda$
(`prepare` default = evidence $\lambda$), the same $\tau$ parametrisations as the samplers (`tau_mode='dz'`, Kz=32 local
windows) and the sampler's colour classes. `test_block_structure_matches_sampler` checks that the classes and their sweep
order coincide with `meta['colors']` of `build_ising_chain` / `build_ising_tree` and that $P$ equals the builder's integer-level
`Qy` after undoing the level scalings. The admissibility fraction is

```math
f=\sum_k\tau_{ik}^2/\sigma_i^2 \quad (\text{exact marginal likelihood requires } f<1).
```

**Gibbs = Gauss-Seidel.** With variables ordered by colour class, $P=D+L+U$ ($D$ diagonal because a class is an independent
set). One deterministic sweep is $x\leftarrow Bx+\text{noise}$ with $B=-(D+L)^{-1}U$ (Amit 1991; Roberts & Sahu 1997; Goodman &
Sokal 1989). Hence

```math
\text{relaxation time } T=\frac{1}{1-\rho(B)},\qquad
\tau_{\rm int}(f)=\frac{g^\top D g}{f^\top P^{-1} f},\ g=P^{-1}f,\qquad
\sup_f\tau_{\rm int}=\frac{1}{\lambda_{\min}(D^{-1/2}PD^{-1/2})}.
```

The IAT formula follows from $\sum_{t\ge0} f^\top B^t P^{-1}f=f^\top P^{-1}(D+L)P^{-1}f$, where only the symmetric part
of $D+L$ survives; so the IAT of a functional does not depend on the scan order, but $\rho$ does. `gs_rho` uses a dense
eigensolve below 2500 variables and ARPACK on the triangular solve `(D+L)^{-1}` above that (3000 variables, 2 s).

**Two-block (data augmentation) rate.** For the blocks $x$ and $z$ the Gibbs chain on $x$ has rate equal to the squared maximal canonical
correlation between $x$ and $z$ (Liu, Wong & Kong 1994):

```math
\rho_{DA}=\lambda_{\max}\!\big(P_{xx}^{-1}P_{xz}P_{zz}^{-1}P_{zx}\big)=1-\lambda_{\min}\!\big(P_{xx}^{-1}\Sigma_{xx}^{-1}\big),
\qquad \frac{1}{1-\rho_{DA}}=\max_v\frac{v^\top P_{xx}v}{v^\top\Sigma_{xx}^{-1}v}
=\max_v \frac{\mathrm{Var}_{\rm post}(v^\top x)}{\mathrm{Var}(v^\top x\mid z)} .
```

$1/(1-\rho_{DA})$ is the inverse of "one minus the fraction of missing information" $F$: the ratio of the posterior
variance of the best-mixing-limited direction to its variance given the auxiliaries.

## 2. Closed-form mechanism

For the chain $P_{xx}=\lambda L+\Lambda$ with $\Lambda_{jj}=\sum_{i\ni j}t_{ij}^2/\tau_{ij}^2$ (each pixel is held by its own
links, so $\Lambda$ is diagonal), while the marginal precision is the true posterior $\Sigma_{xx}^{-1}=\lambda L+T^\top W T$ (compensated
chain; with compensation off the data weight becomes $1/(\sigma^2+S)$). Therefore

```math
\frac{1}{1-\rho_{DA}}=\max_v\frac{v^\top(\lambda L+\Lambda)v}{v^\top(\lambda L+T^\top WT)v}\ \ge\ 1+\max_{v\in\ker T}\frac{v^\top\Lambda v}{\lambda\, v^\top L v}.
```

* **Data directions.** Take $v$ proportional to the chord sum on one chord (uniform $\tau$): the ratio is
  $\mathrm{Var}_{\rm post}(S_i)/\sum_k\tau_{ik}^2$, i.e. $1/(1-F)$ with
  $F=1-(\text{$\tau$-induced variance})/(\text{posterior variance of the partial sum})$ (`chord_sum_rayleigh`). It is $O(1/f)$ and is
  *not* the bottleneck: it is $\le 10^2$.
* **Null directions** (what the data cannot see, held only by the prior) dominate: there the marginal precision is only
  $\lambda v^\top Lv$, but the auxiliaries add $\Lambda_{jj}\propto 1/\tau^2$. The bound is tight (within 3% of the exact $\rho_{DA}$ on every
  case we tried; figure `mixing_rho_vs_lambda.png`). With uniform $\tau_k^2=f\sigma^2/n$ this gives
  $T_{DA}\approx 1+C/(f\lambda)$: fitted on the 16x16 problem (24 chords), $T_{DA}\,f\simeq 8.3\cdot10^2$ for $f\in[0.003,0.99]$
  (4.1e6 -> 1.2e4 for the full GS time over the same range) and $T_{DA}\propto\lambda^{-1}$ over 3 decades.
* **Why compensation does not help the rate.** Compensation changes one diagonal entry per chord (the endpoint precision
  $1/s_i^2$ instead of $1/\sigma_i^2$). The relaxation time changes by at most 2% (e.g. 61154 vs 61275 sweeps at $\tau/\Delta z=0.1$). Its job is
  to keep the *target* unbiased for a given $\tau$; the *rate* is governed by $f$. Tight $\tau$ (small $f$) sends $F\to1$ as
  $1/f$. Once $f\to1$ the endpoint is pinned ($s_i^2\to0$), but that only removes the last degree of freedom of the chord sum.
* **Why a calibrated (softer) prior hurts.** $T_{DA}\propto1/\lambda$. The evidence $\lambda$ is 9x smaller than the
  discrepancy $\lambda$ on this problem (9.75 vs 89.5), so at fixed $f=0.9$ the chain's relaxation time is 13810 vs 1672 sweeps (8.3x; $f=0.3$: 41363 vs 4966). At fixed
  $\tau/\Delta z=0.75$ the gap is only 1179 vs 776 because the evidence-$\lambda$ z windows are wider and push $f$ to 7.4 (vs 1.4), i.e. bias is traded for speed.
  Dense Gibbs suffers the same $1/\lambda$ but with $D_{jj}=\sum_i t_{ij}^2/\sigma_i^2$ in place
  of $\Lambda_{jj}$ (42 vs 11 sweeps for evidence vs discrepancy $\lambda$).
* **Chord length.** $T_{GS}\approx T_{DA}\cdot(n/4)^2$ for the chain (z diffuses along the chord): fitted exponent
  of $T_{GS}$ in $n$ is **1.94** for the chain and **1.23** for the tree (grid sizes 8..28, fixed chords per camera, $f=0.5$,
  `mixing_rho_vs_chord_length.png`), while $T_{DA}$ stays flat and dense Gibbs is flat ($n^{-0.3}$). So the tree
  removes about half the diffusion exponent but not the data-augmentation factor; it is *not* $(\log n)^2$.

## 3. The barrier is structural (proposition)

**Claim.** Let a local embedding have (i) Gaussian auxiliaries private to each chord, (ii) exact marginal likelihood
$N(b_i;t_i^\top x,\sigma_i^2)$ after integrating them out, and (iii) no pixel-pixel $T^\top T$ edges, i.e. the chord's contribution to
$P_{xx}$, $\Lambda_i$, is diagonal. Then

```math
\operatorname{tr}\Lambda_i\ \ge\ \frac{\|t_i\|_1^2}{\sigma_i^2}\ =\ n_{\rm eff,i}\cdot\frac{\|t_i\|_2^2}{\sigma_i^2}
\ =\ n_{\rm eff,i}\cdot\operatorname{tr}\,\mathrm{diag}(t_it_i^\top/\sigma_i^2),\qquad n_{\rm eff}=\|t\|_1^2/\|t\|_2^2\le n .
```

*Proof.* $P_{xx}=\Sigma_{xx}^{-1}+P_{xz}P_{zz}^{-1}P_{zx}\succeq\Sigma_{xx}^{-1}$ and integrating out chord $i$'s private
auxiliaries returns the factor $t_it_i^\top/\sigma_i^2$, so $\Lambda_i\succeq t_it_i^\top/\sigma_i^2$. For diagonal $\Lambda_i$ test with $w=\Lambda_i^{-1}t_i$:
$t^\top\Lambda^{-1}t\ge (t^\top\Lambda^{-1}t)^2/\sigma^2$, so $\sum_k t_k^2/\Lambda_{kk}\le\sigma^2$. Cauchy-Schwarz,
$(\sum_k|t_k|)^2\le(\sum_k t_k^2/\Lambda_{kk})(\sum_k\Lambda_{kk})$, gives the claim. $\square$

Numerically (16x16, $f=0.99$): $\operatorname{tr}\Lambda=2.24\cdot10^5\ge 1.90\cdot10^5=\sum_i\|t_i\|_1^2/\sigma_i^2$, versus
$1.66\cdot10^4$ for dense Gibbs, a factor 11 (mean $n=15$). `remedies.json: trace_bound`.

*Consequence.* A diagonal conditional precision $n_{\rm eff}$ times larger than dense Gibbs's own (which already pays $D/(\lambda\ell_{\ker})$ in prior-held directions) means the data-augmentation
factor satisfies roughly $T_{DA}\gtrsim n_{\rm eff}\,T_{\rm dense}$ in any directions where the data are weak. We measure $T_{DA}/T_{\rm dense}=40$ vs $n_{\rm eff}/f=23$ (16x16, $f=0.5$). This is a statement about *every*
such embedding: tree allocation of $\tau_v$ only redistributes the same budget $\sum_v\tau_v^2\le\sigma^2$ and the leaf-level
pixel terms see at best $\sim2\sigma^2/n$, so I-tree cannot escape (consistent with the measurement: its $T_{DA}$ is
1.6x *larger* than the chain's). The only free parameters are $\tau$ (exactly the $f$ family) and the endpoint variance (irrelevant to the rate).

Assumptions that make this an honest scope statement: Gaussian relaxation of the discrete model; chord-private auxiliaries;
diagonal $\Lambda$ (bounded degree, no pixel-pixel edges); deterministic single-site-type updates; $f<1$ for exactness.
Outside the proposition: raising the degree (grouping), non-exact targets ($f>1$), and different update rules (over-relaxation).

## 4. Validation

Tiny problem (12x12, 24 chords), default config ($\tau/\Delta z=0.75$, Kz=32, compensated, evidence $\lambda$). IAT in sweeps of the slowest
pixel functional 'worst' (exact sup over pixel functionals), 'total' (summed emissivity), 'pc1' (top posterior PC), 'centre' pixel.
Theory = `iat_functional`; continuous = 128 stationary chains x 40000 exact linear Gibbs sweeps; discrete = 4 chains x 40000
sweeps of `sample_chain`/`sample_tree` (8 x 40000 for the dense Ising sampler); thin 10 for chain/tree.

| model | functional | theory | continuous sim | discrete Ising |
|---|---|---:|---:|---:|
| dense | centre / total / pc1 / worst | 7.0 / 61 / 75 / 93 | 7.1 / 62 / 76 / 92 | 4.4 / 9.7 / 12.7 / 13.1 |
| chain | centre / total / pc1 / worst | 121 / 1435 / 1708 / 1837 | 97 / 1313 / 1462 / 1549 | 35 / 255 / 263 / 570 |
| tree | centre / total / pc1 / worst | 209 / 1760 / 2211 / 2340 | 177 / 1806 / 2027 / 2176 | 102 / 317 / 374 / 381 |

(`mixing_predicted_vs_measured.png`.) Linear theory matches the exact simulation to within the Monte-Carlo error (5-20% for the slow
functionals). The `test_rho_matches_simulated_decay` and `test_iat_formula_matches_simulation` unit tests do the same on random sparse Gaussians.

**The discrete samplers are 5-7x *faster* than the continuous prediction at $\tau/\Delta z=0.75$**, uniformly over dense, chain and tree. The
discrete target is truncated to $[0,\varepsilon_{\max}]$ with K=8 levels, so its posterior spread in the slow directions is
2-5x smaller than the Gaussian's (e.g. total: sd 0.11 vs 0.24, worst: 0.12 vs 0.40), which shortens the relaxation. The *ratio* to dense is preserved to within a factor of about 2: chain/dense 26-44 measured vs 20-24 predicted, tree/dense 29-33 vs 25-29.
So the theory predicts the slowdown factor of the construction, not the absolute IAT of the truncated model.
The README's "hundreds of sweeps" is the discrete number; the continuous theory predicts thousands.

**Rigidity ($\tau<\Delta z$) reverses the sign of the gap** (`mixing_rigidity.png`, chain, 40000 sweeps x 4 chains):

| $\tau/\Delta z$ | theory total / worst | discrete total / worst | frozen pixels, R-hat med |
|---:|---:|---:|---:|
| 0.25 | 12 486 / 15 692 | >6.2e4 / >8.0e4 (lower bound) | 26%, 1.79 |
| 0.5 | 3 169 / 3 976 | 3.0e4 / 6.3e4 (lower bound) | 2%, 1.02 |
| 0.75 | 1 435 / 1 837 | 327 / 874 | 0, 1.001 |
| 1.5 | 379 / 524 | 94 / 152 | 0, 1.000 |

Below $\tau/\Delta z\approx0.6$ the lattice is rigid and the discrete chain is 10-16x slower than the continuous prediction (and
glassy: 26% of pixels never move at 0.25); above it the truncation makes it 4-5x faster. The run-to-run spread of the 0.75 row (255 to 327 for 'total') indicates 30-50% IAT uncertainty.

**Default 32x32 problem** (806 pixels, 72 chords, 3020/2948 joint variables for chain/tree, sparse eigensolver):

| model | $\rho$ | $1/(1-\rho)$ | slowest-mode IAT | $\rho_{DA}$ | $f$ (unclamped) |
|---|---:|---:|---:|---:|---:|
| dense | 0.97175 | 35 | 78 | - | - |
| chain | 0.999824 | 5 676 | 11 351 | 0.9965 | 6.4 |
| tree | 0.999709 | 3 440 | 6 903 | 0.9987 | 3.1 |

With the default schedule (2000 warm-up + 5000 sweeps) the linear theory predicts that the continuous chain is not converged. Hardware cost per sample is sweeps x colour blocks:
dense 35 x 315 = 1.1e4, chain 5676 x 76 = 4.3e5, tree 3440 x 114 = 3.9e5 block updates, i.e. about 35x more even though each block is local (dense is not
implementable on bounded-degree hardware, so this is the price of locality in the Gaussian theory; discrete-model measurements put the sweep ratio at 26-44x).

## 5. Remedies

All numbers on 16x16 (24 chords), evidence $\lambda$, dense reference $T=42$ (`mixing_remedies.png`, `mixing_rho_vs_tau.png`).

| remedy | preserves target? | local / bounded degree? | predicted effect | verdict |
|---|---|---|---|---|
| larger $\tau$ with compensation up to $f\to1$ | yes (exact for $f<1$) | yes | $T\propto1/f$: 4.1e6 ($f=.003$) -> 1.3e4 (0.9) -> 1.26e4 (0.99) | saturates at $T\approx1.3\cdot10^4$ (chain), $6.5\cdot10^3$ (tree): 300x / 150x dense |
| $\tau$ beyond the limit ($f>1$, what the default $\tau/\Delta z=0.75$ does on calibrated problems) | **no**: chord variance inflated by $1+f$ | yes | $T\approx1.2\cdot10^3$ at $f=7.4$, 125 at $f=118$ | trades bias for speed: $T\cdot(\text{chord variance}/\sigma^2)\approx\text{const}\approx10^4$ (conservation law: it is the tempered likelihood) |
| tempering the auxiliaries (scale link energies by $1/\Theta$) | only with compensation | yes | identical to $\tau\to\sqrt\Theta\tau$, i.e. the same $f$ family | nothing new |
| stiffer prior $\lambda$ | no (changes the posterior) | yes | $T\propto1/\lambda$ | not allowed for calibration; discrepancy $\lambda$ gives an 8x speed-up at fixed $f$ and a 6-10x too stiff posterior (README) |
| non-centred auxiliaries (residuals $e_k=z_k-z_{k-1}-t_kx_k$ as the variables) | yes | **no**: the data term $(b-\sum t x-\sum e)^2/s^2$ couples all $n$ pixels of a chord | becomes dense Gibbs ($T\approx42$) | the $g=n$ limit of the next row; not bounded degree |
| group $g$ pixels per auxiliary | yes | degree grows $\propto g$ | $T$: 24831 (g=1, degree 16), 6523 (2, 14), 1803 (4, 20), 481 (8, 41), 229 (12, 63), 65 (16, 74) | $T\approx T_1/g^{1.9}$ (diffusion $(n/g)^2$ plus a nearly constant $\rho_{DA}$ until $g\approx n$); degree 20 / 41 / 74 (1.3x / 2.6x / 4.6x that of g=1) buys 14x / 52x / 380x |
| over-relaxed (SOR) Gibbs, $x_c\leftarrow(1-\omega)x_c+\omega\mu_c+\sqrt{\omega(2-\omega)}\,\sigma_c\xi$ | yes (invariant for $\omega\in(0,2)$; Adler 1981, Barone-Frigessi 1990) | yes (single-site, same graph) | chain $T=2.5\cdot10^4\to1.0\cdot10^3$ at $\omega\approx1.95$; tree $1.3\cdot10^4\to1.8\cdot10^3$ at $\omega\approx1.7$; dense gets *worse* (42 at 1) | ~20x / ~7x for the chain / tree, but still 25-45x slower than dense; requires a continuous or ordered-categorical conditional (Neal's ordered over-relaxation), which a binary p-bit does not provide |

Interpretation. The auxiliary construction costs $T\simeq n_{\rm eff}/f\times T_{\rm dense}$ in the data-augmentation factor and a further $n^2$ (chain) or $\approx n^{1.2}$ (tree) from diffusion
of the partial sums. For bounded degree $d$, grouping is the only target-preserving knob and it follows roughly
$T\approx T_{DA}(n/g)^2$ with $d\propto g$.

## 6. Conclusions for the paper

1. For a Gaussian relaxation, colour-block Gibbs is Gauss-Seidel on $P$; $\rho(B)$ and the IAT of any functional are exactly
   computable from sparse linear algebra, and agree with simulated autocorrelation (5-20%).
2. The slowdown of I-chain/I-tree relative to dense Gibbs is the data-augmentation factor $1/(1-\rho_{DA})=\max_v \mathrm{Var}_{\rm post}(v^\top x)/\mathrm{Var}(v^\top x\mid z)$.
   It is set by the null directions of $T$: $T_{DA}\approx1+C/(f\lambda)$, growing like $1/\tau^2$, $1/\lambda$ (softer prior is worse) and, through the
   chain diffusion, like $n^2$ (chain) or $n^{1.2}$ (tree).
3. Compensation does not change the rate (at most 2%); it only decouples $\tau$ from bias. The attainable $\tau$ is capped by $f<1$.
4. The barrier is structural: any exact, chord-private, bounded-degree Gaussian embedding has
   $\operatorname{tr}\Lambda\ge n_{\rm eff}\operatorname{tr}\mathrm{diag}(T^\top WT)$, hence $T_{DA}\gtrsim n_{\rm eff}T_{\rm dense}$ in prior-held directions.
5. The discrete samplers differ from the Gaussian theory by a constant factor (5-7x faster at $\tau/\Delta z=0.75$ because of truncation; 10-16x slower below $\tau/\Delta z\approx0.6$ because of lattice rigidity), but the *relative* slowdown to dense is predicted within a factor of ~2.
6. No hardware-local remedy we examined closes the gap. Over-relaxation is the best target-preserving change (20x for the chain) but still leaves
   a 25x gap and needs a non-binary conditional; grouping trades degree for speed at $\propto g^{-2}$; inflating $\tau$ beyond $f=1$ is a likelihood-tempering bias.

**Caveats.** The theory is for the continuous Gaussian relaxation; the discrete numbers come from one tiny problem (12x12, 8 chords/camera, peaked phantom, seed 0) with 30-50% IAT uncertainty and, for the slow cases, a run length comparable to the IAT. Chord-length scaling varies the grid at fixed chords per camera, so pixel size changes with $n$. The $\rho$ of SOR for dense Gibbs reflects a non-consistently-ordered colouring (many classes).

## References (verified to exist)

* Y. Amit, "On rates of convergence of stochastic relaxation for Gaussian and non-Gaussian distributions", *J. Multivariate Analysis* 38(1):82-99, 1991.
  Related: Y. Amit and U. Grenander, "Comparing sweep strategies for stochastic relaxation", *J. Multivariate Analysis* 37(2):197-222, 1991.
* G. O. Roberts and S. K. Sahu, "Updating schemes, correlation structure, blocking and parameterization for the Gibbs sampler", *J. R. Stat. Soc. B* 59(2):291-317, 1997.
* J. S. Liu, W. H. Wong and A. Kong, "Covariance structure of the Gibbs sampler with applications to the comparisons of estimators and augmentation schemes", *Biometrika* 81(1):27-40, 1994.
* J. Goodman and A. D. Sokal, "Multigrid Monte Carlo method. Conceptual foundations", *Phys. Rev. D* 40:2035, 1989 (not re-checked online).
* S. L. Adler, "Over-relaxation method for the Monte Carlo evaluation of the partition function for multiquadratic actions", *Phys. Rev. D* 23:2901, 1981; D. Barone and A. Frigessi, "Improving stochastic relaxation for Gaussian random fields", *Probab. Engrg. Inform. Sci.* 4:369-389, 1990 (not re-checked online); R. M. Neal, "Suppressing random walks in MCMC using ordered over-relaxation" (1998) (not re-checked online).

The three statistics citations (Amit, Roberts-Sahu, Liu-Wong-Kong) were located via web search (IDEAS/RePEc and publisher listings). The
statement that the two-component rate equals the squared maximal canonical correlation is the standard result of Liu, Wong & Kong; the
page-level theorem numbers were not checked.
