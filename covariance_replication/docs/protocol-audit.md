# protocol audit

checked against the supplied replication protocol and the september 2023 manuscript, *graph-based methods for forecasting realized covariances*, with the authors' released scripts as an additional reference. the audit applies to the notebook in this folder.

[comparison with the quoted empirical claims](result-comparison.md)

## overall assessment

the covariance methodology is implemented, with documented differences in the data source and forecast period. the saved public experiment is an adapted replication. the original numerical benchmark requires the original realized covariance panel or licensed synchronized quotes.

## requirement checks

| protocol requirement | implementation and verification |
| --- | --- |
| 27 paper stocks | all 27 tickers match appendix a, including its frozen sector assignments |
| lobster inputs, july 2007–july 2021 | optional reader-supplied minute-quote csv adapter. the saved run uses public syncster trade bars starting in 2016 |
| original out-of-sample window | differs. the public panel yields 623 one-session origins from july 2022 to december 2024, in 30 monthly blocks |
| 09:30–16:00 sampling | utc timestamps are converted to new york time, preserving daylight-saving changes. at least 350 minute bars are required and carry-forward is limited to ten minutes within the same session |
| five-minute subsampling | covariance and quarticity are averaged across five offset grids. analytic regression checks verify both calculations |
| realized covariance and drd | full 27 × 27 covariance matrices, diagonal variances, and 351 pairwise correlations |
| daily, weekly and monthly history | disjoint lag groups 1, 2–5 and 6–22, with regression checks for the exact offsets |
| variance graphs | empty, complete, appendix-a sectors and graphical lasso. 5-fold cv uses only past returns. graph diagonal is zero and normalization is symmetric |
| correlation graphs | empty, complete and line graph of all 351 pairs. pair adjacency means sharing a stock. regression checks compare the efficient aggregation with an explicit normalized adjacency |
| har-cholesky and har-drd | both included alongside all eleven graph-enhanced combinations |
| ordinary least squares | per-series intercepts and shared slopes. scaled pivoted qr avoids forming normal equations. a rank-deficient regression is checked analytically |
| 1,000-observation calibration | 1,000 retained feature dates before each forecast month. incomplete forward labels are purged, leaving 996 and 979 labels at the 5- and 22-session horizons |
| positive covariance forecasts | correlations are clipped to [-1,1]. statsmodels cov_nearest performs the clipped repair with an explicit 1e-4 correlation eigenvalue floor. variance diagonals are preserved. all scored predictions pass cholesky factorization |
| covariance losses | euclidean includes each unique entry and the diagonal. frobenius counts both off-diagonal halves. matrix qlike uses log determinant and trace of the solved system. checked against small hand-calculated matrices |
| model confidence set | sequential sq with 10,000 stationary bootstrap draws, mean block length 2 and 5% exclusion level. metric-specific probabilities are shown. longer block lengths are also checked for overlapping horizons |
| gmvp and gmvp+ | unconstrained linear solve and cvxpy/osqp long-only quadratic program. analytic diagonal-covariance weights check both solutions |
| risk and turnover | the one-session exercise uses the protocol's annualized standard deviation and drift-adjusted turnover. returns are retained-session open-to-close simple returns, without overnight exposure or transaction costs |
| market stress | spy realized variance above the full-panel 90th percentile. ratios and mcs are reported separately for calm and stress dates |
| quarticity robustness | main run follows raw-quarticity interactions printed in manuscript equations 17–18. a separate same-fold run uses the protocol's square-root-quarticity interaction |

## conflicts within the reference material

the protocol labels horizons as 4 and 21 while summing from zero through h minus one. the manuscript's equation 15 sums from zero through h and uses h=4 and h=21. we follow the manuscript, giving 5 and 22 retained sessions.

the manuscript prints raw quarticity in its harq and gharq interactions. the protocol specifies square-root quarticity. both are now evaluated, with separate harq-drd denominators, so their results are not mixed.

the line graph is formed from all stock pairs, as in section 3.4.3 of the manuscript. restricting it to the selected glasso stock edges would change the set of correlation series and would not implement that model.

the protocol's example mcs p-values do not describe every loss column. the manuscript gives distinct probabilities for euclidean, frobenius and qlike. the notebook computes all three separately and treats the published numbers as reference observations.

## limits of the public data

the 1,647-session intersection has calendar gaps. lags and forecast targets count retained observations, so five observed sessions can span more than one exchange week. the public experiment cannot establish the original paper's precise calendar-horizon results. keeping all tickers and the 1,000-observation calibration was the chosen adaptation.

the public trade-bar feed differs from lobster midpoint quotes in sampling, market coverage and potential corporate-action treatment. screening for valid prices, synchronized times and finite positive variances does not establish that the feed reproduces lobster measurements.

the eigenvalue floor is explicit and higher than machine precision. repairs can therefore affect inverse-covariance scores and portfolio weights. longer-horizon portfolio statistics use separate nonoverlapping cohorts and annualize by retained holding sessions.

the original 120-month lobster experiment, the authors' exact data preprocessing and appendix-only variants such as daily/weekly refits remain unverified.

## reproduce the checks

from this folder after installing requirements:

```sh
python -m unittest discover -s tests -v
```

then execute the notebook from this folder. its outputs record forecast coverage, calibration boundaries, graph diagnostics, all losses, both quarticity choices, and portfolio performance.
