# Extended Results: Adaptive Covariance Shape Forecasting

This extension studies covariance **shape** — the unit-trace positive-definite kernel that determines fully invested minimum-variance portfolio weights independently of total-variance scaling. Its common total-variance forecast comes from HAR-DRD. The three shape experts are HAR-DRD, GHAR(GL,L) and HAR-Cholesky; exponential weights update only after a target is fully observed.

[Executed extension](../Extended_Covariance.ipynb) · [Complete proofs](theory.md) · [Fixed design](extension-design.json) · [Saved tables](tables)

---

## 1  Mathematical Foundations

The extension is grounded in five formal results that connect graph spectral theory, online learning and covariance geometry. Each result below states its assumptions and practical scope.

### Lemma 1 — Line-Graph Aggregation and Spectral Contraction

Let $`n \ge 4`$ and $`B \in \{0,1\}^{n \times m}`$ be the endpoint incidence matrix for $`n`$ stocks and $`m = n(n-1)/2`$ pairs. The normalized line-graph adjacency is

```math
W_L = \frac{B^\top B - 2I_m}{2(n-2)},
```

with eigenvalues (and multiplicities)

```math
1\;(1),\quad \frac{n-4}{2(n-2)}\;(n-1),\quad -\frac{1}{n-2}\;\left(\tfrac{n(n-3)}{2}\right).
```

For any pair-feature matrix $`X \in \mathbb{R}^{m \times d}`$ and the row-mean projector $`J_m = \mathbf{1}\mathbf{1}^\top / m`$,

```math
\|W_L^k X - J_m X\|_F \le \rho_n^k \|X - J_m X\|_F, \quad \rho_n = \max\!\left\{\frac{|n-4|}{2(n-2)},\; \frac{1}{n-2}\right\} < 1.
```

| parameter | value for this study |
| --- | --- |
| stocks $`n`$ | 27 |
| pairs $`m`$ | 351 |
| contraction rate $`\rho_{27}`$ | **0.46** per pass |
| effective pair neighbors | 50 out of 350 |

**Practical significance.** Repeated application of the fixed linear line-graph operator contracts centered pair features by at most $`0.46^k`$ after $`k`$ passes. Three passes therefore leave at most **9.73%** of the initial centered norm. This is an operator bound, rather than an observed reduction or an optimal-depth claim for a nonlinear graph network. The identity $`(W_L X)_{(i,j)} = \sum_{k \ne i,j}[X_{(i,k)} + X_{(j,k)}] / [2(n-2)]`$ shows that aggregation averages the other-partner correlations of each endpoint, providing a principled smoothing mechanism for pair-specific noise. The factorization $`B^\top(BX)`$ avoids materializing the $`351 \times 351`$ adjacency and runs in $`O(n^2 d)`$.

### Proposition 2 — Loss–Risk Counterexample

There exist positive-definite forecasts $`A`$ and $`C`$ and a true conditional covariance $`H`$ such that $`C`$ has strictly smaller Euclidean, Frobenius **and** QLIKE loss relative to $`H`$, yet the GMVP constructed from $`C`$ has **higher** true portfolio variance than the GMVP from $`A`$.

| quantity | forecast $`A = 2H`$ | forecast $`C`$ | interpretation |
| --- | --- | --- | --- |
| $`H`$ | $`\text{diag}(1,4)`$ | $`\text{diag}(1,4)`$ | true covariance |
| Euclidean / Frobenius | $`\sqrt{17} \approx 4.123`$ | **1.000** | $`C`$ wins all three losses |
| excess QLIKE | 0.3863 | **0.0457** | $`C`$ wins QLIKE |
| GMVP weights | $`(4/5,\; 1/5)`$ | $`(3/4,\; 1/4)`$ | different allocations |
| true GMVP variance | **0.800** | 0.8125 | but $`A`$'s portfolio is safer |

**Why this happens.** GMVP weights are invariant to positive scaling of the covariance ($`w_{cH} = w_H`$ for all $`c > 0`$), yet absolute matrix-distance losses still penalize the scalar offset. This non-monotone relationship between estimation accuracy and portfolio risk persists across all three standard loss functions and both unconstrained and long-only constraints. It is therefore **insufficient** to infer allocation improvement from matrix-loss improvement alone — a separate portfolio evaluation is mandatory.

### Lemma 3 — Conditional Portfolio-Risk Bound via Relative Spectral Error

If a forecast $`S \succ 0`$ satisfies the relative spectral sandwich

```math
(1 - \delta) H \preceq S \preceq (1 + \delta) H, \quad 0 \le \delta < 1,
```

then for any GMVP (unrestricted or long-only) under matching constraints,

```math
w_S^\top H w_S \;\le\; \frac{1 + \delta}{1 - \delta}\, w_H^\top H w_H.
```

| relative error $`\delta`$ | risk inflation factor | excess risk bound |
| ---: | ---: | ---: |
| 0.01 | 1.0202 | +2.02% |
| 0.05 | 1.1053 | +10.53% |
| 0.10 | 1.2222 | +22.22% |
| 0.20 | 1.5000 | +50.00% |

**Practical scope.** This bound requires a **relative** (multiplicative) spectral approximation, not merely low average absolute error. It bridges the gap identified in Proposition 2 by providing an explicit condition under which estimation quality does translate to portfolio quality. This relative condition has not been verified for the latent conditional covariance in the public experiment.

### Lemma 4 — Bounded Shape Loss and Positive Mixtures

For trace-one positive-semidefinite shapes $`C`$ and $`C^*`$,

```math
\ell(C, C^*) = \tfrac{1}{2}\|C - C^*\|_F^2 \in [0,\, 1].
```

Convex mixtures $`C_p = \sum_j p_j C_j`$ satisfy the **convexity inequality**

```math
\ell(C_p, C^*) \le \sum_j p_j \,\ell(C_j, C^*),
```

and if each expert $`C_j \succ 0`$, then $`\hat{S} = \tau \cdot C_p`$ is positive definite with trace $`\tau`$ for any $`\tau > 0`$. This ensures that the adaptive mixture preserves the covariance structure needed for downstream optimization.

### Theorem 5 — Delayed-Feedback Regret Bound

Fix $`K`$ experts, learning rate $`\eta > 0`$, and an arbitrary sequence of trace-one realized shapes. If $`d_t`$ unresolved labels are outstanding at origin $`t`$, the cumulative adaptive mixture incurs

```math
\sum_{t=1}^T \ell(C_{p_t}, C_t^*) - \min_j \sum_{t=1}^T \ell_{t,j} \;\le\; \frac{\log K}{\eta} + \frac{\eta T}{8} + \frac{\eta}{4}\sum_{t=1}^T d_t.
```

| horizon $`h`$ | feedback | bound form | value at $`K = 3`$, $`\eta = 0.5`$ |
| ---: | --- | --- | ---: |
| 1 session | immediate | $`\log K / \eta`$ (tighter, via concavity) | **2.197** |
| 5 sessions | delayed up to 4 | general with delay term | 598.3 |
| 22 sessions | delayed up to 21 | general with delay term | 2,781.8 |

**Corollary (one-session).** When feedback is immediate and $`\eta \le 1/2`$, the loss exponential is concave on the trace-one domain. Jensen's inequality replaces the bounded-moment bound, eliminating the $`\eta T / 8`$ term. At $`K = 3`$ and $`\eta = 0.5`$, the regret is bounded by $`2 \log 3 \approx 2.197`$ **regardless of the sequence length**. This is the strongest of the three horizon-specific bounds.

**Practical context.** The general delayed bound at 22 sessions (2,781.8 for 1,045 origins) exceeds one loss unit per origin and is therefore vacuous at that horizon. The sharper constant bound at $`h = 1`$ is the operationally useful result.

---

## 2  Experimental Protocol

### Design

| element | specification |
| --- | --- |
| **development period** | July 2022 – December 2024 (623 one-session origins) |
| **confirmation period** | January 2025 – October 2026 (443 one-session origins) |
| **split timing** | method, experts and $`\eta = 0.5`$ fixed after inspecting development; before running confirmation |
| **primary endpoint** | adaptive vs. uniform one-session normalized shape loss |
| **secondary endpoints** | 5- and 22-session shape loss; raw matrix losses; portfolio risk |
| **bootstrap** | 10,000 stationary-bootstrap draws, seeded; mean block max(5, $`h`$) for shape, max(2, $`h`$) for MCS |

This local design record is not an external preregistration. Forecasts and mixture weights continue to update causally during confirmation.

### Expert Construction

The three experts are computed from the same synchronized five-minute return covariances. Each expert's unit-trace shape $`C_j = S_j / \text{tr}(S_j)`$ preserves its relative eigenvalue structure while removing the total-variance dimension. The common scale $`\tau_t = \text{tr}(\hat{S}_t^{\text{HAR-DRD}})`$ is applied to the mixture shape, yielding the final forecast $`\hat{S}_t = \tau_t \cdot C_{p_t}`$.

---

## 3  Confirmation Results

### 3.1  Primary Shape-Loss Comparison

The confirmation sample contains **443 one-session forecasts**. The primary endpoint — adaptive versus uniform one-session normalized shape loss — yields:

| statistic | value |
| --- | --- |
| paired mean difference | **−3.21 × 10⁻⁵** |
| 95% bootstrap CI | [−7.02 × 10⁻⁵, +4.42 × 10⁻⁶] |
| **interval includes zero** | **yes** |

Five-session (CI: [−3.16 × 10⁻⁵, +9.26 × 10⁻⁶]) and 22-session (CI: [−4.77 × 10⁻⁵, +7.57 × 10⁻⁵]) secondary comparisons also include zero. **These comparisons do not establish a directional confirmation improvement.**

### 3.2  Development vs. Confirmation Dynamics

| period | $`h`$ | origins | adaptive mean | uniform mean | difference | 95% lower | 95% upper |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| development | 1 | 623 | 0.02190 | 0.02192 | −2.19 × 10⁻⁵ | −3.17 × 10⁻⁵ | −1.22 × 10⁻⁵ |
| confirmation | 1 | 443 | 0.02475 | 0.02478 | −3.21 × 10⁻⁵ | −7.02 × 10⁻⁵ | +4.42 × 10⁻⁶ |
| development | 5 | 619 | 0.01069 | 0.01071 | −1.96 × 10⁻⁵ | −2.54 × 10⁻⁵ | −1.44 × 10⁻⁵ |
| confirmation | 5 | 439 | 0.01270 | 0.01271 | −1.27 × 10⁻⁵ | −3.16 × 10⁻⁵ | +9.26 × 10⁻⁶ |
| development | 22 | 602 | 0.00867 | 0.00870 | −2.31 × 10⁻⁵ | −3.48 × 10⁻⁵ | −1.27 × 10⁻⁵ |
| confirmation | 22 | 422 | 0.01199 | 0.01199 | −2.33 × 10⁻⁷ | −4.77 × 10⁻⁵ | +7.57 × 10⁻⁵ |

**Key observation:** Development intervals are entirely negative at all three horizons; confirmation intervals all include zero. The development gains are not confirmed by these intervals. These comparisons alone do not identify whether the change comes from overfitting, market regimes or other sample differences.

### 3.3  Raw Matrix Losses (Confirmation, $`h = 1`$)

| model | Euclidean ratio | Frobenius ratio | QLIKE ratio | Euclidean MCS $`p`$ | Frobenius MCS $`p`$ | QLIKE MCS $`p`$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Adaptive shape** | **0.9977** | **0.9979** | **0.9918** | 1.000 | 1.000 | 1.000 |
| Uniform shape | 0.9982 | 0.9987 | 0.9926 | 0.377 | 0.527 | <0.001 |
| HAR-DRD | 1.0000 | 1.0000 | 1.0000 | 0.377 | 0.527 | <0.001 |
| GHAR(GL,L) · common trace | 1.0012 | 0.9992 | 1.2181 | 0.355 | 0.527 | <0.001 |
| GHAR(GL,L) | 1.0113 | 1.0109 | 1.1609 | 0.355 | 0.527 | <0.001 |
| HAR-Cholesky | 1.0137 | 1.0062 | 1.0764 | 0.355 | 0.527 | <0.001 |
| HAR-Cholesky · common trace | 1.0167 | 1.0189 | 1.0036 | 0.055 | 0.063 | <0.001 |

Ratios use HAR-DRD as the denominator; lower is better. The adaptive mixture has the **lowest one-session raw matrix losses** across all seven specifications. It is the **sole model retained** in the QLIKE MCS at the 5% level. Its Euclidean and Frobenius MCS sets retain other models, including HAR-DRD — the QLIKE dominance does not extend to distance losses.

### 3.4  Portfolio Performance (Confirmation)

| model | GMVP annual risk (%) | GMVP+ annual risk (%) | GMVP turnover | GMVP+ turnover |
| --- | ---: | ---: | ---: | ---: |
| 1/N (equal) | 12.596 | — | 0.010 | — |
| HAR-DRD | 9.326 | 9.435 | 0.458 | 0.329 |
| HAR-Cholesky | **9.170** | **9.405** | 0.437 | 0.308 |
| Uniform shape | 9.221 | 9.466 | 0.415 | 0.305 |
| Adaptive shape | 9.228 | 9.491 | 0.415 | 0.307 |
| GHAR(GL,L) | 30.162 | 9.747 | 0.770 | 0.336 |

**Key finding.** The adaptive mixture's unrestricted GMVP risk (9.228%) is lower than HAR-DRD (9.326%) but higher than HAR-Cholesky (9.170%) and the uniform mixture (9.221%). Its long-only risk (9.491%) exceeds all three comparators. Learning the mixture weights therefore **does not establish a portfolio gain** in confirmation. This result is consistent with Proposition 2: matrix-loss superiority need not translate to allocation superiority.

### 3.5  Conditioning and Stability Diagnostics

| model | median $`\kappa`$ | max $`\kappa`$ | max gross leverage | max leverage date | min eigenvalue |
| --- | ---: | ---: | ---: | --- | ---: |
| HAR-DRD | 38.00 | 157.23 | 2.07 | 2025-04-14 | 0.2151 |
| GHAR(GL,L) | 33.12 | **127,174** | **37.05** | 2025-04-11 | 0.0017 |
| HAR-Cholesky | 41.06 | 247.61 | 2.54 | 2025-04-10 | 0.1581 |
| Uniform shape | 36.42 | 218.93 | 2.30 | 2025-04-14 | 0.2157 |
| Adaptive shape | 35.70 | 223.37 | 2.32 | 2025-04-11 | 0.2137 |

The raw GHAR(GL,L) forecast reaches condition number 127,174 and gross leverage 37.05. The adaptive mixture's maximum condition number is 223.37, approximately **570 times smaller**. These diagnostics expose inverse sensitivity and extreme positions; they do not isolate the cause of each realized loss or establish a universal conditioning guarantee for shape mixing. The separately designed spectral guard has its own explicit bound; see [Proposition 6](theory.md#proposition-6-spectral-conditioning-and-gross-leverage).

---

## 4  Cumulative Regret Analysis

| horizon | origins | adaptive cumulative | best-expert cumulative | **regret** | theoretical upper bound |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,066 | 24.608 | 24.688 | **−0.080** | 2.197 |
| 5 | 1,062 | 12.252 | 12.284 | −0.032 | 598.3 |
| 22 | 1,045 | 10.370 | 10.234 | +0.136 | 2,781.8 |

**At one session**, the adaptive mixture's cumulative shape loss is 24.608, versus 24.688 for the best fixed expert over the full 2022–2026 run. Its regret of **−0.080** is not only within the theoretical bound of 2.197 but is **negative** — meaning the adaptive learner outperformed the best fixed expert in hindsight. This is a realized performance statement over the complete learning trajectory; it is not an independently reset confirmation guarantee.

**At 22 sessions**, regret is positive (+0.136) and the theoretical bound (2,781.8) is vacuous. The delay penalty $`(\eta/4) \sum_t d_t`$ dominates when unresolved labels accumulate, as 21,714 total delayed observations inflate the bound far beyond practical informativeness.

---

## 5  What Changed After Development

Development favored the raw graph model over HAR-DRD on short-horizon matrix losses. That ordering **reverses** for one-session confirmation losses. A fixed uniform combination already captures most of the mixture's descriptive advantage. Evaluating only the development period would give an **overly favorable impression** of the incremental value of adaptive weights.

### Causal Weight Dynamics

The exponential-weights rule uses only cumulative observed shape losses. It has no explicit condition-number or volatility-regime input. The saved weight plot shows the realized path, but the tables above do not establish a causal regime explanation or a rule that admits the graph expert only at moderate condition numbers. Those claims require a separate analysis.


---

## 6  Use Cases

| application | what this study provides |
| --- | --- |
| **Forecast pooling** | Combine distinct covariance estimates while preserving positive definiteness and a chosen trace; retain the uniform mixture as a simple, strong control for the value of learning. |
| **Risk-model review** | Distinguish lower covariance error from lower allocation risk using a reproducible counterexample (Proposition 2), raw matrix losses, and both portfolio constraint sets. |
| **Pair-network design** | Compute stock-pair message aggregation through endpoint sums and quantify how repeated aggregation reduces centered contrasts at rate $`\rho_{27}^k = 0.46^k`$. |
| **Delayed targets** | Release multi-session labels only after the final retained session, preventing unresolved targets from influencing earlier mixture weights. The formal delayed-feedback bound governs this causal constraint. |
| **Research replication** | Compare the same method across development and confirmation periods and report when the paper's model ranking fails to carry over — a template for honest method evaluation. |
| **Regularization via mixing** | The observed maximum condition number is about 570 times smaller than the raw graph model's. This is a sample diagnostic, rather than a general conditioning guarantee. |

These are **research uses**. This experiment does not validate a live allocation policy or include execution costs.

---

## 7  Scope, Reproducibility and Limitations

### Data

The data are pinned public Syncster minute trade bars for the manuscript's 27 stocks and SPY. Unit-trace covariance shapes require synchronized multivariate intraday returns, which daily univariate volatility series cannot supply. The original study's licensed LOBSTER quote panel is unavailable here; the loader also supports locally prepared licensed minute inputs. The feed, years, price construction and retained calendar differ from the original experiment.

### Calendar

The strict shared-calendar rule removes sessions, particularly before 2022, so a model horizon counts retained sessions. A 1,000-observation calibration can span more than four exchange years.

### Statistical Methodology

- All six shape intervals (3 horizons × 2 periods) are **pointwise**, without a simultaneous multiple-testing guarantee.
- Stationary-bootstrap draws are seeded with 10,000 replications; the primary interval's mean block length is max(5, $`h`$), and MCS uses max(2, $`h`$).
- MCS is a sequential elimination procedure; a single paired comparison and a multi-model elimination procedure answer different statistical questions.

### Reproducibility

Run both notebooks from the `covariance_replication` folder with its requirements. The extension saves primary shape comparisons, all horizon-specific matrix/MCS tables, portfolio results, cumulative regret and conditioning diagnostics in `docs/tables`.

[Replication audit](protocol-audit.md) · [Paper comparison](result-comparison.md)
