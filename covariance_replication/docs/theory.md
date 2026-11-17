# mathematical extension and complete proofs

this note establishes properties of the covariance replication and motivates an adaptive extension. the spectral identity and exponential-weights method have established antecedents. these are self-contained specializations and a project-specific counterexample; mathematical priority is unassessed.

## notation

let $`n\ge4`$ be the number of stocks and $`m=n(n-1)/2`$ the number of distinct correlation pairs. covariance forecasts are symmetric positive definite, realized covariances are nonzero positive semidefinite, and portfolio weights sum to one. $`\|\cdot\|_F`$ is the frobenius norm and $`\preceq`$ denotes positive-semidefinite order.

## lemma 1: exact line-graph aggregation and contraction

let $`B\in\{0,1\}^{n\times m}`$ have a one at each of the two stock endpoints of every pair. the normalized adjacency of the correlation line graph is

```math
W_L=\frac{B^\top B-2I_m}{2(n-2)}.
```

for any matrix $`X\in\mathbb{R}^{m\times d}`$ of pair features,

```math
(W_LX)_{(i,j)}
=\frac{\sum_{k\ne i,j}X_{(i,k)}+\sum_{k\ne i,j}X_{(j,k)}}{2(n-2)}.
```

its eigenvalues, with multiplicities, are

```math
1\ (1),\qquad \frac{n-4}{2(n-2)}\ (n-1),
\qquad-\frac1{n-2}\ \left(\frac{n(n-3)}2\right).
```

if $`J_m=\mathbf{1}\mathbf{1}^\top/m`$, then

```math
\|W_L^kX-J_mX\|_F\le
\rho_n^k\|X-J_mX\|_F,\quad
\rho_n=\max\left\{\frac{|n-4|}{2(n-2)},\frac1{n-2}\right\}<1.
```

**proof.**

the diagonal of $`B^\top B`$ is two. distinct pairs have inner product one exactly when they share a stock. each pair has $`2(n-2)`$ neighbors, proving both formulas. also $`BB^\top=(n-2)I_n+\mathbf{1}\mathbf{1}^\top`$, whose eigenvalues are $`2(n-1)`$ once and $`n-2`$ on the remaining $`n-1`$ dimensions. transferring the nonzero eigenvalues to $`B^\top B`$ and accounting for its $`m-n`$ zero eigenvalues gives the stated spectrum. $`W_L`$ is symmetric, its constant eigenvector has eigenvalue one, and its restriction to the orthogonal complement has operator norm $`\rho_n`$. apply the spectral theorem to each feature column. ∎

for this project, $`n=27`$ and $`\rho_n=0.46`$. repeating the same linear pair aggregation contracts centered pair features by at most $`0.46^k`$. this explains a property of the graph operator. it does not establish that a particular empirical forecasting depth is optimal.

the expression $`B^\top(BX)`$ also permits aggregation in $`O(n^2d)`$ operations without materializing an $`m\times m`$ adjacency. [spectral reference](https://aeb.win.tue.nl/2WF02/spectra.pdf)

## proposition 2: covariance loss reductions can coexist with higher portfolio risk

there are forecasts with smaller euclidean, frobenius and matrix qlike losses that nevertheless yield a higher true gmvp variance.

**proof by construction.**

let the true conditional covariance be $`H=\operatorname{diag}(1,4)`$ and compare

```math
A=2H=\operatorname{diag}(2,8),\qquad C=\operatorname{diag}(1,3).
```

both are positive definite. the euclidean and frobenius distances to $`H`$ are $`\sqrt{17}`$ for $`A`$ and $`1`$ for $`C`$. define excess matrix qlike as $`Q(S,H)-Q(H,H)`$, where $`Q(S,H)=\log|S|+\operatorname{tr}(S^{-1}H)`$. its values are

```math
Q(A,H)-Q(H,H)=2\log2-1\approx0.386294,
```

```math
Q(C,H)-Q(H,H)=\log(3/4)+1/3\approx0.045651.
```

the forecast based on $`C`$ therefore has smaller loss on all three metrics. however,

```math
w_A=(4/5,1/5),\quad w_C=(3/4,1/4),
```

```math
w_A^\top Hw_A=4/5=0.8,\qquad
w_C^\top Hw_C=13/16=0.8125.
```

both portfolios are long-only, so the construction applies to gmvp and gmvp+. ∎

the reason is that gmvp weights are invariant to a positive scalar multiple of the entire covariance matrix. absolute matrix distance still penalizes that scalar error. this construction shows why the portfolio conclusion needs its own empirical test.

## lemma 3: a relative spectral condition controls oracle portfolio risk

let $`H\succ0`$ and suppose a forecast $`S\succ0`$ satisfies

```math
(1-\delta)H\preceq S\preceq(1+\delta)H,\qquad 0\le\delta<1.
```

let $`w_H`$ minimize $`w^\top Hw`$ and $`w_S`$ minimize $`w^\top Sw`$ over the same feasible set, either all fully invested portfolios or the long-only simplex. then

```math
w_S^\top Hw_S\le
\frac{1+\delta}{1-\delta}\,w_H^\top Hw_H.
```

**proof.**

positive-semidefinite ordering and optimality give

```math
(1-\delta)w_S^\top Hw_S
\le w_S^\top Sw_S
\le w_H^\top Sw_H
\le(1+\delta)w_H^\top Hw_H.
```

divide by $`1-\delta`$. ∎

the condition is a relative matrix bound. an improvement in average absolute covariance error alone does not imply it. the theorem makes no assertion that the public forecasts satisfy this condition for the latent conditional covariance.

## lemma 4: bounded covariance-shape loss and positive forecasts

define a covariance shape $`C=S/\operatorname{tr}(S)`$ and a realized shape $`C^*=H/\operatorname{tr}(H)`$. if both are positive semidefinite with trace one, then

```math
\ell(C,C^*)=\tfrac12\|C-C^*\|_F^2\in[0,1].
```

for probability weights $`p_j`$, define $`C_p=\sum_jp_jC_j`$. then

```math
\ell(C_p,C^*)\le\sum_jp_j\ell(C_j,C^*).
```

if each $`C_j\succ0`$ and a forecast-time scale $`\tau>0`$ is available, $`\hat S=\tau C_p`$ is positive definite and has trace $`\tau`$.

**proof.**

each trace-one positive-semidefinite matrix has nonnegative eigenvalues summing to one, so its squared frobenius norm is at most one. also $`\operatorname{tr}(CC^*)\ge0`$. expanding the squared distance proves the upper bound of two before multiplication by one half. convexity of the squared norm gives the mixture inequality. for any nonzero $`x`$, $`x^\top C_px=\sum_jp_jx^\top C_jx>0`$. trace linearity proves the final claim. ∎

the implemented extension uses three experts: har-drd, ghar(gl,l) and har-cholesky. it retains the har-drd trace $`\tau_t`$ and learns the mixture of their unit-trace shapes. a uniform shape mixture and each expert under the same trace are included as controls.

## theorem 5: delayed-feedback regret bound for the shape mixture

fix $`K`$ positive-definite expert forecasts per origin and $`\eta>0`$. at origin $`t`$, let $`O_t`$ contain the earlier origins whose realized targets are fully observed. set

```math
p_{t,j}=
\frac{\exp(-\eta\sum_{s\in O_t}\ell_{s,j})}
{\sum_{i=1}^K\exp(-\eta\sum_{s\in O_t}\ell_{s,i})},
```

where $`\ell_{s,j}=\ell(C_{s,j},C_s^*)`$. let $`d_t`$ be the number of earlier unresolved labels. using the mixture of lemma 4,

```math
\sum_{t=1}^T\ell(C_{p_t},C_t^*)
-\min_j\sum_{t=1}^T\ell_{t,j}
\le
\frac{\log K}{\eta}+\frac{\eta T}{8}
+\frac{\eta}{4}\sum_{t=1}^Td_t.
```

this is a deterministic bound for any admissible expert and realized-matrix sequence. its comparator is the best fixed expert under unit-trace shape loss over the same origins.

**proof.**

first consider the hypothetical full-feedback weights

```math
q_{t,j}\propto\exp(-\eta\sum_{s<t}\ell_{s,j}).
```

for a random variable $`L\in[0,1]`$ under weights $`q_t`$, the bounded exponential-moment inequality is

```math
\log E[e^{-\eta L}]\le-\eta E[L]+\eta^2/8.
```

one derivation writes $`f(s)=\log E[e^{sL}]`$. under the tilted distribution, $`f''(s)`$ is a variance bounded by $`1/4`$. integrating the second derivative around zero gives $`f(s)\le sf'(0)+s^2/8`$.

apply this inequality to the exponential-weights normalizer and telescope over origins. its final value is at least $`K^{-1}\exp(-\eta\min_j\sum_t\ell_{t,j})`$. this gives

```math
\sum_t q_t^\top\ell_t-\min_j\sum_t\ell_{t,j}
\le\log K/\eta+\eta T/8.
```

to account for the delayed labels, put $`u_j=\sum_{s<t,\ s\notin O_t}\ell_{s,j}\in[0,d_t]`$ and interpolate

```math
r_j(a)\propto p_{t,j}e^{-a\eta u_j},\qquad 0\le a\le1.
```

then $`r(0)=p_t`$ and $`r(1)=q_t`$, and

```math
\frac{d}{da}\sum_jr_j(a)\ell_{t,j}
=-\eta\,\operatorname{Cov}_{r(a)}(\ell_t,u).
```

by cauchy–schwarz and the bounded-range variance inequality, its absolute value is at most $`\eta d_t/4`$. integrate over $`a`$ and sum over $`t`$. finally apply lemma 4 to replace weighted expert loss by mixture loss. ∎

for an $`h`$-session target, the code makes the label available only after its final session is complete. with at most one forecast per retained session, $`d_t\le h-1`$. the notebook uses $`\eta=0.5`$ fixed before inspecting the extension results. the bound concerns normalized shape loss. it does not imply a reduction in raw matrix losses or realized portfolio risk. [exponential-weights antecedent](https://www.sciencedirect.com/science/article/pii/S002200009791504X)

### corollary: constant regret for the one-session mixture

with immediate feedback, $`0<\eta\le1/2`$ and the same trace-one shape domain,

```math
\sum_t\ell(C_{p_t},C_t^*)-\min_j\sum_t\ell_{t,j}
\le\frac{\log K}{\eta}.
```

**proof.**

for a fixed realized trace-one shape $`C^*`$, put $`f(C)=\exp[-\eta\|C-C^*\|_F^2/2]`$. in vector coordinates its hessian is

```math
\nabla^2 f(C)=\eta f(C)
\left[\eta\,\operatorname{vec}(C-C^*)\operatorname{vec}(C-C^*)^\top-I\right].
```

lemma 4 gives $`\|C-C^*\|_F^2\le2`$, so this hessian is negative semidefinite for $`\eta\le1/2`$. hence $`f`$ is concave on the convex trace-one covariance domain. jensen's inequality gives

```math
e^{-\eta\ell(C_{p_t},C_t^*)}
\ge\sum_jp_{t,j}e^{-\eta\ell_{t,j}}.
```

the logarithm of the right side is the change in the exponential-weights normalizer. telescope and use the best expert's contribution to that normalizer to obtain the bound. ∎

at $`K=3`$ and the implemented $`\eta=0.5`$, the one-session cumulative regret bound is $`2\log3\approx2.1972`$ in normalized shape-loss units. the longer-horizon forecasts use the general delayed bound above.

## proposition 6: spectral conditioning and gross leverage

let the positive eigenvalues of a covariance forecast have minimum $`m`$, maximum $`M`$ and mean $`\mu`$. fix a condition-number limit $`\kappa>1`$. define

```math
e=(M-\kappa m)_+,\qquad a=\frac{e}{e+(\kappa-1)\mu},\qquad \lambda_i'=(1-a)\lambda_i+a\mu.
```

then all cleaned eigenvalues are positive, their sum is unchanged, and their condition number is at most $`\kappa`$.

**proof.**

because $`\mu>0`$, the coefficient satisfies $`0\le a<1`$. each cleaned eigenvalue is therefore a convex combination of positive values. summing over eigenvalues proves trace preservation.

if $`M\le\kappa m`$, then $`e=a=0`$ and the original spectrum already satisfies the limit. otherwise,

```math
(1-a)(M-\kappa m)-a(\kappa-1)\mu=0.
```

therefore the largest cleaned eigenvalue minus $`\kappa`$ times the smallest is zero. in both cases the required ratio is at most $`\kappa`$. any subsequent common positive scaling preserves this ratio. ∎

**gross-leverage consequence.** let $`S\succ0`$ have condition number at most $`\kappa`$, and let $`w`$ be its unrestricted fully invested minimum-variance portfolio. then

```math
\|w\|_1\le\sqrt{\kappa}.
```

**proof.**

write $`m_S`$ and $`M_S`$ for the smallest and largest eigenvalues of $`S`$. optimality relative to the equal-weight portfolio gives

```math
m_S\|w\|_2^2\le w^\top Sw\le\left(\frac{\mathbf{1}}{n}\right)^\top S\left(\frac{\mathbf{1}}{n}\right)\le\frac{M_S}{n}.
```

cauchy–schwarz gives $`\|w\|_1\le\sqrt{n}\|w\|_2\le\sqrt{M_S/m_S}\le\sqrt{\kappa}`$. ∎

at the implemented limit $`\kappa=100`$, unrestricted gross leverage is at most 10 in exact arithmetic. long-only fully invested portfolios have gross leverage one. these are conditional properties of the forecast construction and its optimizer; they do not guarantee lower realized portfolio risk.
