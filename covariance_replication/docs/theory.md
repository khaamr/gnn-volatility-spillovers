# Mathematical Extension and Complete Proofs

This note establishes properties of the covariance replication and motivates an adaptive extension. The spectral identity and exponential-weights method have established antecedents. These are self-contained specializations and a project-specific counterexample; mathematical priority is unassessed.

## Notation

Let $n \ge 4$ be the number of stocks and $m = n(n-1)/2$ the number of distinct correlation pairs. Covariance forecasts are symmetric positive definite, realized covariances are nonzero positive semidefinite, and portfolio weights sum to one. $\|\cdot\|_F$ is the Frobenius norm and $\preceq$ denotes positive-semidefinite order.

---

## Lemma 1: Exact Line-Graph Aggregation and Contraction

Let $B \in \{0,1\}^{n \times m}$ have a one at each of the two stock endpoints of every pair. The normalized adjacency of the correlation line graph is

$$W_L = \frac{B^\top B - 2I_m}{2(n-2)}.$$

For any matrix $X \in \mathbb{R}^{m \times d}$ of pair features,

$$(W_L X)_{(i,j)} = \frac{\displaystyle\sum_{k \ne i,j} X_{(i,k)} + \sum_{k \ne i,j} X_{(j,k)}}{2(n-2)}.$$

Its eigenvalues, with multiplicities, are

$$1 \;\;(1), \qquad \frac{n-4}{2(n-2)} \;\;(n-1), \qquad -\frac{1}{n-2} \;\;\left(\frac{n(n-3)}{2}\right).$$

If $J_m = \mathbf{1}\mathbf{1}^\top / m$, then

$$\|W_L^k X - J_m X\|_F \le \rho_n^k \|X - J_m X\|_F, \quad \rho_n = \max\left\{\frac{|n-4|}{2(n-2)},\; \frac{1}{n-2}\right\} < 1.$$

**Proof.**
The diagonal of $B^\top B$ is two. Distinct pairs have inner product one exactly when they share a stock. Each pair has $2(n-2)$ neighbors, proving both formulas. Also $B B^\top = (n-2)I_n + \mathbf{1}\mathbf{1}^\top$, whose eigenvalues are $2(n-1)$ once and $n-2$ on the remaining $n-1$ dimensions. Transferring the nonzero eigenvalues to $B^\top B$ and accounting for its $m - n$ zero eigenvalues gives the stated spectrum. $W_L$ is symmetric, its constant eigenvector has eigenvalue one, and its restriction to the orthogonal complement has operator norm $\rho_n$. Apply the spectral theorem to each feature column. $\blacksquare$

For this project, $n = 27$ and $\rho_n = 0.46$. Repeating the same linear pair aggregation contracts centered pair features by at most $0.46^k$. This explains a property of the graph operator. It does not establish that a particular empirical forecasting depth is optimal.

The expression $B^\top(BX)$ also permits aggregation in $O(n^2 d)$ operations without materializing an $m \times m$ adjacency. [Spectral reference](https://aeb.win.tue.nl/2WF02/spectra.pdf)

---

## Proposition 2: Covariance Loss Reductions Can Coexist with Higher Portfolio Risk

There are forecasts with smaller Euclidean, Frobenius and matrix QLIKE losses that nevertheless yield a higher true GMVP variance.

**Proof by construction.**
Let the true conditional covariance be $H = \operatorname{diag}(1, 4)$ and compare

$$A = 2H = \operatorname{diag}(2, 8), \qquad C = \operatorname{diag}(1, 3).$$

Both are positive definite. The Euclidean and Frobenius distances to $H$ are $\sqrt{17}$ for $A$ and $1$ for $C$. Define excess matrix QLIKE as $Q(S, H) - Q(H, H)$, where $Q(S, H) = \log|S| + \operatorname{tr}(S^{-1}H)$. Its values are

$$Q(A, H) - Q(H, H) = 2\log 2 - 1 \approx 0.386294,$$

$$Q(C, H) - Q(H, H) = \log(3/4) + 1/3 \approx 0.045651.$$

The forecast based on $C$ therefore has smaller loss on all three metrics. However,

$$w_A = (4/5,\; 1/5), \qquad w_C = (3/4,\; 1/4),$$

$$w_A^\top H w_A = 4/5 = 0.8, \qquad w_C^\top H w_C = 13/16 = 0.8125.$$

Both portfolios are long-only, so the construction applies to GMVP and GMVP+. $\blacksquare$

The reason is that GMVP weights are invariant to a positive scalar multiple of the entire covariance matrix. Absolute matrix distance still penalizes that scalar error. This construction shows why the portfolio conclusion needs its own empirical test.

---

## Lemma 3: A Relative Spectral Condition Controls Oracle Portfolio Risk

Let $H \succ 0$ and suppose a forecast $S \succ 0$ satisfies

$$(1 - \delta) H \preceq S \preceq (1 + \delta) H, \qquad 0 \le \delta < 1.$$

Let $w_H$ minimize $w^\top H w$ and $w_S$ minimize $w^\top S w$ over the same feasible set, either all fully invested portfolios or the long-only simplex. Then

$$w_S^\top H w_S \le \frac{1 + \delta}{1 - \delta} \, w_H^\top H w_H.$$

**Proof.**
Positive-semidefinite ordering and optimality give

$$(1 - \delta) w_S^\top H w_S \le w_S^\top S w_S \le w_H^\top S w_H \le (1 + \delta) w_H^\top H w_H.$$

Divide by $1 - \delta$. $\blacksquare$

The condition is a relative matrix bound. An improvement in average absolute covariance error alone does not imply it. The theorem makes no assertion that the public forecasts satisfy this condition for the latent conditional covariance.

---

## Lemma 4: Bounded Covariance-Shape Loss and Positive Forecasts

Define a covariance shape $C = S / \operatorname{tr}(S)$ and a realized shape $C^* = H / \operatorname{tr}(H)$. If both are positive semidefinite with trace one, then

$$\ell(C, C^*) = \tfrac{1}{2}\|C - C^*\|_F^2 \in [0, 1].$$

For probability weights $p_j$, define $C_p = \sum_j p_j C_j$. Then

$$\ell(C_p, C^*) \le \sum_j p_j \,\ell(C_j, C^*).$$

If each $C_j \succ 0$ and a forecast-time scale $\tau > 0$ is available, $\hat{S} = \tau C_p$ is positive definite and has trace $\tau$.

**Proof.**
Each trace-one positive-semidefinite matrix has nonnegative eigenvalues summing to one, so its squared Frobenius norm is at most one. Also $\operatorname{tr}(C C^*) \ge 0$. Expanding the squared distance proves the upper bound of two before multiplication by one half. Convexity of the squared norm gives the mixture inequality. For any nonzero $x$, $x^\top C_p x = \sum_j p_j x^\top C_j x > 0$. Trace linearity proves the final claim. $\blacksquare$

The implemented extension uses three experts: HAR-DRD, GHAR(GL,L) and HAR-Cholesky. It retains the HAR-DRD trace $\tau_t$ and learns the mixture of their unit-trace shapes. A uniform shape mixture and each expert under the same trace are included as controls.

---

## Theorem 5: Delayed-Feedback Regret Bound for the Shape Mixture

Fix $K$ positive-definite expert forecasts per origin and $\eta > 0$. At origin $t$, let $O_t$ contain the earlier origins whose realized targets are fully observed. Set

$$p_{t,j} = \frac{\exp\!\left(-\eta \displaystyle\sum_{s \in O_t} \ell_{s,j}\right)}{\displaystyle\sum_{i=1}^K \exp\!\left(-\eta \sum_{s \in O_t} \ell_{s,i}\right)},$$

where $\ell_{s,j} = \ell(C_{s,j}, C_s^*)$. Let $d_t$ be the number of earlier unresolved labels. Using the mixture of Lemma 4,

$$\sum_{t=1}^T \ell(C_{p_t}, C_t^*) - \min_j \sum_{t=1}^T \ell_{t,j} \le \frac{\log K}{\eta} + \frac{\eta T}{8} + \frac{\eta}{4} \sum_{t=1}^T d_t.$$

This is a deterministic bound for any admissible expert and realized-matrix sequence. Its comparator is the best fixed expert under unit-trace shape loss over the same origins.

**Proof.**
First consider the hypothetical full-feedback weights

$$q_{t,j} \propto \exp\!\left(-\eta \sum_{s < t} \ell_{s,j}\right).$$

For a random variable $L \in [0, 1]$ under weights $q_t$, the bounded exponential-moment inequality is

$$\log \mathbb{E}[e^{-\eta L}] \le -\eta\, \mathbb{E}[L] + \eta^2 / 8.$$

One derivation writes $f(s) = \log \mathbb{E}[e^{sL}]$. Under the tilted distribution, $f''(s)$ is a variance bounded by $1/4$. Integrating the second derivative around zero gives $f(s) \le s f'(0) + s^2/8$.

Apply this inequality to the exponential-weights normalizer and telescope over origins. Its final value is at least $K^{-1} \exp(-\eta \min_j \sum_t \ell_{t,j})$. This gives

$$\sum_t q_t^\top \ell_t - \min_j \sum_t \ell_{t,j} \le \frac{\log K}{\eta} + \frac{\eta T}{8}.$$

To account for the delayed labels, put $u_j = \sum_{s < t,\; s \notin O_t} \ell_{s,j} \in [0, d_t]$ and interpolate

$$r_j(a) \propto p_{t,j}\, e^{-a \eta u_j}, \qquad 0 \le a \le 1.$$

Then $r(0) = p_t$ and $r(1) = q_t$, and

$$\frac{d}{da} \sum_j r_j(a)\, \ell_{t,j} = -\eta \operatorname{Cov}_{r(a)}(\ell_t, u).$$

By Cauchy–Schwarz and the bounded-range variance inequality, its absolute value is at most $\eta\, d_t / 4$. Integrate over $a$ and sum over $t$. Finally apply Lemma 4 to replace weighted expert loss by mixture loss. $\blacksquare$

For an $h$-session target, the code makes the label available only after its final session is complete. With at most one forecast per retained session, $d_t \le h - 1$. The notebook uses $\eta = 0.5$ fixed before inspecting the extension results. The bound concerns normalized shape loss. It does not imply a reduction in raw matrix losses or realized portfolio risk. [Exponential-weights antecedent](https://www.sciencedirect.com/science/article/pii/S002200009791504X)

### Corollary: Constant Regret for the One-Session Mixture

With immediate feedback, $0 < \eta \le 1/2$ and the same trace-one shape domain,

$$\sum_t \ell(C_{p_t}, C_t^*) - \min_j \sum_t \ell_{t,j} \le \frac{\log K}{\eta}.$$

**Proof.**
For a fixed realized trace-one shape $C^*$, put $f(C) = \exp[-\eta \|C - C^*\|_F^2 / 2]$. In vector coordinates its Hessian is

$$\nabla^2 f(C) = \eta\, f(C) \left[\eta \operatorname{vec}(C - C^*) \operatorname{vec}(C - C^*)^\top - I\right].$$

Lemma 4 gives $\|C - C^*\|_F^2 \le 2$, so this Hessian is negative semidefinite for $\eta \le 1/2$. Hence $f$ is concave on the convex trace-one covariance domain. Jensen's inequality gives

$$e^{-\eta\, \ell(C_{p_t}, C_t^*)} \ge \sum_j p_{t,j}\, e^{-\eta\, \ell_{t,j}}.$$

The logarithm of the right side is the change in the exponential-weights normalizer. Telescope and use the best expert's contribution to that normalizer to obtain the bound. $\blacksquare$

At $K = 3$ and the implemented $\eta = 0.5$, the one-session cumulative regret bound is $2 \log 3 \approx 2.1972$ in normalized shape-loss units. The longer-horizon forecasts use the general delayed bound above.

---

## Proposition 6: Spectral Conditioning and Gross Leverage

Let the positive eigenvalues of a covariance forecast have minimum $m$, maximum $M$ and mean $\mu$. Fix a condition-number limit $\kappa > 1$. Define

$$e = (M - \kappa\, m)_+, \qquad a = \frac{e}{e + (\kappa - 1)\mu}, \qquad \lambda_i' = (1 - a)\lambda_i + a\mu.$$

Then all cleaned eigenvalues are positive, their sum is unchanged, and their condition number is at most $\kappa$.

**Proof.**
Because $\mu > 0$, the coefficient satisfies $0 \le a < 1$. Each cleaned eigenvalue is therefore a convex combination of positive values. Summing over eigenvalues proves trace preservation.

If $M \le \kappa\, m$, then $e = a = 0$ and the original spectrum already satisfies the limit. Otherwise,

$$(1 - a)(M - \kappa\, m) - a(\kappa - 1)\mu = 0.$$

Therefore the largest cleaned eigenvalue minus $\kappa$ times the smallest is zero. In both cases the required ratio is at most $\kappa$. Any subsequent common positive scaling preserves this ratio. $\blacksquare$

**Gross-leverage consequence.** Let $S \succ 0$ have condition number at most $\kappa$, and let $w$ be its unrestricted fully invested minimum-variance portfolio. Then

$$\|w\|_1 \le \sqrt{\kappa}.$$

**Proof.**
Write $m_S$ and $M_S$ for the smallest and largest eigenvalues of $S$. Optimality relative to the equal-weight portfolio gives

$$m_S \|w\|_2^2 \le w^\top S w \le \left(\frac{\mathbf{1}}{n}\right)^\top S \left(\frac{\mathbf{1}}{n}\right) \le \frac{M_S}{n}.$$

Cauchy–Schwarz gives $\|w\|_1 \le \sqrt{n}\, \|w\|_2 \le \sqrt{M_S / m_S} \le \sqrt{\kappa}$. $\blacksquare$

At the implemented limit $\kappa = 100$, unrestricted gross leverage is at most 10 in exact arithmetic. Long-only fully invested portfolios have gross leverage one. These are conditional properties of the forecast construction and its optimizer; they do not guarantee lower realized portfolio risk.
