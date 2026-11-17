# spectral architecture results

[executed notebook](../../Spectral_Covariance.ipynb) · [design and acceptance rule](design.json) · [architecture and primary sources](research.md) · [saved tables](tables)

the geometric spectral gate improves both point estimates against har-drd on the reused 2025–october 2026 benchmark, but **does not pass the joint uncertainty criterion**. the covariance-error result is stronger than the earlier shape mixture. the portfolio evidence remains inconclusive.

## equal-priority comparison

there are 443 shared one-session benchmark origins. all annualized risk figures below are percentages; portfolios are long-only and exclude transaction costs.

| model | euclidean ratio | frobenius ratio | qlike ratio | annual risk (%) |
| --- | ---: | ---: | ---: | ---: |
| har-drd | 1.0000 | 1.0000 | 1.0000 | 9.4351 |
| previous adaptive shape | 0.9977 | 0.9979 | 0.9918 | 9.4912 |
| uniform shape | 0.9982 | 0.9987 | 0.9926 | 9.4664 |
| har-cholesky | 1.0137 | 1.0062 | 1.0764 | 9.4047 |
| arithmetic har covariance | 1.0827 | 1.0906 | 1.0158 | 9.2113 |
| log-euclidean har covariance | 0.9860 | 0.9808 | 1.0897 | 9.4763 |
| static geometric pool | **0.9766** | **0.9740** | 0.9999 | 9.4587 |
| geometric spectral gate | 0.9839 | 0.9821 | 0.9961 | 9.4061 |

the dynamic gate has lower risk than the newly trained static control; arithmetic har covariance and har-cholesky have lower risk in the wider table. matrix ratios use har-drd as denominator. the dynamic gate has better euclidean and frobenius results than the previous adaptive mixture, but its qlike ratio is worse. there is no all-metric dominance claim.

## uncertainty

| dynamic gate versus reference | frobenius change (%) | paired 95% interval | long-only variance change (%) | paired 95% interval |
| --- | ---: | --- | ---: | --- |
| har-drd | −1.7889 | [−2.8223, −0.6606] | −0.6131 | [−2.1492, 0.9510] |
| previous adaptive shape | −1.5800 | [−3.1139, −0.1895] | −1.7840 | [−3.6196, 0.9258] |
| uniform shape | −1.6636 | [−3.2978, −0.2076] | −1.2689 | [−2.6499, 0.9121] |
| static geometric pool | +0.8330 | [−0.1297, 2.1910] | −1.1089 | [−2.1592, 0.2970] |

risk changes in this table refer to **variance**, rather than standard deviation. the annualized standard deviation falls from 9.4351% to 9.4061% against har-drd, a decrease of 0.0290 percentage points, or about 0.307% relatively.

the joint rule requires the upper ends of both primary intervals against har-drd to be below zero. the risk interval includes zero, so the candidate fails that rule. comparisons against additional controls are exploratory. mean block length is 10 and there are 10,000 seeded stationary-bootstrap draws. the separate multi-model confidence-set table retains har-drd for frobenius loss; a single paired comparison and a multi-model elimination procedure answer different questions.

## why the static control matters

the static pool uses seven trainable parameters and achieves lower euclidean and frobenius point losses than the 175-parameter dynamic network. the network has lower long-only risk than that static pool. the simpler arithmetic covariance pool has the lowest risk in the reported table but substantially higher matrix losses. these results expose a trade-off rather than a universal best model.

the 2024 development evaluation contains 245 origins. the dynamic gate lowers frobenius distance by 6.65% against har-drd, while long-only variance is 0.62% higher. the change in ranking across development and the reused benchmark reinforces the need for a fresh evaluation sample.

## guaranteed properties and implementation checks

the covariance construction guarantees positive definiteness and condition number at most 100. the benchmark maximum is 100 up to floating-point precision. this also gives a worst-case unrestricted gross-leverage bound of 10. the guarantee concerns the forecast matrix and weights, rather than realized risk.

training uses 40 differentiable accelerated projected-gradient steps for long-only weights. final evaluation uses exact constrained optimization. on the reused benchmark, the mean relative forecast-variance objective gap is 0.00140%, and the maximum is 0.01991%. the largest l1 difference between approximate and exact weights is 0.03527, so the training optimizer is explicitly treated as an approximation.

the saved fold audit verifies that each refit label predates the forecast origin and that each inner validation label follows the fitting block. structural tests check positive definiteness, conditioning, simplex feasibility, permutation equivariance conditional on the expert inputs, finite training gradients and absence of future realized covariance in expert construction.

## interpretation

the literature search identified a useful extension: geometric covariance experts combined with learned spectral cleaning and an explicitly risk-aware objective. on this reused benchmark it improves the average accuracy–risk pair against har-drd and the earlier adaptive mixture. uncertainty prevents a definite two-endpoint improvement claim. the architecture was not retuned after inspecting these benchmark results.

fresh dates or an independently specified universe are required to assess confirmation after this search. seed sensitivity, transaction costs, additional horizons and isolation of the gate versus cleaner contributions remain untested. the public-data and retained-calendar limitations of the original replication also remain.
