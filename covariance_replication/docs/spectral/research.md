# geometric pooling with learned spectral cleaning

this extension searches for a model that improves both covariance forecasts and long-only portfolio risk. the earlier common-trace shape mixture produced mixed results. its uniform control already captured most gains, while the raw graph forecast occasionally became ill-conditioned. the new architecture therefore learns from current historical regimes, allows total variance to change, and explicitly cleans eigenvalues.

[executed notebook](../../Spectral_Covariance.ipynb) · [fixed design](design.json) · [implementation](../../spectral.py) · [results](results.md)

## literature search

| primary research | relevant architectural idea | implication for this project |
| --- | --- | --- |
| [bucci, palma and zhang: geometric deep learning for realized covariance matrix forecasting](https://arxiv.org/html/2412.09517v1) | spd-aware regression and geometric daily, weekly and monthly summaries | add a log-euclidean covariance expert alongside an arithmetic control |
| [bongiorno, manolakis and mantegna: end-to-end large portfolio optimization through covariance cleaning](https://arxiv.org/html/2507.01918v3) | joint temporal, volatility and eigenvalue processing trained for realized portfolio risk | include a risk objective and spectral cleaning rather than assuming matrix accuracy implies better portfolios |
| [yang, zhao and chai: neural nonlinear shrinkage](https://arxiv.org/html/2601.15597v1) | a lightweight neural eigenvalue transformation applied to an established covariance estimator | learn a shared spectral map while retaining positive covariance eigenvalues |
| [tapia costa, cucuringu and nason: higher order dynamic network linear models](https://doi.org/10.2139/ssrn.5113698) | higher-order neighborhoods and time-varying volatility graphs | a relevant alternative for a future graph ablation; not implemented in this experiment |

the geometric study reports covariance forecasting gains on a different 50-stock lobster sample. the two covariance-cleaning studies report portfolio gains under their own protocols. these findings motivate the candidate; they do not establish its performance on our 27-stock trade-bar panel. the 2026 neural-shrinkage paper is a preprint. this implementation is a new combination for the project, rather than an exact reproduction of any searched paper or a claim of mathematical priority.

## implemented architecture

1. construct five positive-definite experts: har-drd, ghar(gl,l), har-cholesky, an arithmetic daily/weekly/monthly covariance pool, and a log-euclidean pool. the graph expert is the source of stock/pair network information.
2. standardize ten historical summary features inside each training fold. a small nonlinear gate assigns five expert weights and a bounded positive total-variance adjustment.
3. pool the full covariance matrices with convex weights. unlike the previous shape mixture, this stage does not fix total variance to har-drd.
4. use a shared neural eigenvalue cleaner to shrink each eigenvalue toward the pooled mean. renormalize the spectrum to retain its trace, then enforce a condition-number limit of 100.
5. train with equal normalized frobenius-distance and squared long-only portfolio-return objectives. approximate constrained weights are differentiable during training; final evaluation uses the exact quadratic-program solver.

the static control learns constant expert weights, a constant variance scale and a scalar shrinkage intensity. it is deliberately simpler. the dynamic candidate adds both regime-dependent weights and nonlinear spectral cleaning; this comparison does not isolate the separate effect of each added component.

## causal training and acceptance rule

the five experts update daily from available inputs; the original har regressions retain monthly fits. gate parameters refit quarterly on at most 504 earlier origins. at least 378 past labels are required. the final 126 historical labels are reserved for inner validation. select among 0, 5, 10, 20 and 40 epochs by minimizing the larger validation ratio for frobenius distance and the long-only mean-square-return proxy, relative to har-drd. refit the selected epoch count on the full historical window. only one fixed random seed is used; seed sensitivity remains untested.

forecast accuracy and portfolio risk are co-primary. the predefined joint benchmark criterion requires both paired 95% intervals against har-drd to lie strictly below zero. additional controls include har-cholesky, the static pool and the previous adaptive and uniform shape mixtures. results are reported even when the candidate loses. alternative endpoints and controls are exploratory.

2024 provides development evaluation. january 2025–october 2026 is a reused retrospective benchmark because we already inspected those outcomes for the previous extension. chronological model fits remain causal, but architectural research has reused the benchmark. fresh dates or an independently specified universe are needed for a new confirmation claim.

## what is guaranteed

the positive eigenvalue construction guarantees a positive-definite output. let the cleaned positive spectrum have maximum m_upper, minimum m_lower and mean mu. define excess = max(m_upper − 100 m_lower, 0), and a = excess / (excess + 99 mu). the final spectrum is (1 − a) times the old spectrum plus a times mu. algebra gives a condition number at most 100, while its trace stays fixed.

for any spd matrix with condition number at most kappa, unrestricted fully invested gmvp weights satisfy gross leverage at most sqrt(kappa). optimality versus equal weights bounds their squared euclidean norm by kappa/n; cauchy–schwarz gives the result. at kappa = 100 the worst-case gross bound is 10. long-only weights have gross leverage one by construction.

these guarantees concern the matrix and feasible weights. they provide no deterministic guarantee of lower future forecast loss or portfolio variance.
