# extended results and use cases

this extension studies covariance **shape**, the information that determines fully invested minimum-variance portfolio weights. its common total-variance forecast comes from har-drd. the three shape experts are har-drd, ghar(gl,l) and har-cholesky; exponential weights update only after a target is fully observed.

[executed extension](../Extended_Covariance.ipynb) · [complete proofs](theory.md) · [fixed design](extension-design.json) · [saved tables](tables)

## mathematical results

| statement | what is established | practical use |
| --- | --- | --- |
| lemma 1 | exact line-graph aggregation; centered features contract by at most 0.46 per pass for 27 stocks | understand smoothing of pair-specific signals and compute pair aggregation without a dense pair-by-pair adjacency |
| proposition 2 | all three covariance losses can improve while true gmvp variance rises from 0.8 to 0.8125 | require a separate portfolio evaluation before claiming an allocation benefit |
| lemma 3 | a relative spectral error bound implies an oracle portfolio-risk bound under matching constraints | state the extra condition needed to turn matrix approximation into a portfolio guarantee |
| lemma 4 | trace-one shape loss is bounded; convex mixtures preserve positive definiteness and the chosen trace | combine forecasts into a covariance matrix suitable for optimization |
| theorem 5 and corollary | finite-sequence shape-loss regret bounds, including delayed feedback; one-session regret at most 2 log 3 | update models causally and compare cumulative normalized error with the best fixed expert |

these are proved specializations of established graph and online-learning ideas, plus an explicit counterexample for this project. mathematical priority is unassessed. the bounds concern the stated loss and conditions; they do not guarantee lower trading risk.

## confirmation experiment

the method, experts and learning rate 0.5 were fixed after inspecting development results from july 2022–december 2024 and before running confirmation outcomes from january 2025–october 2026. this local design record is not an external preregistration. forecasts and mixture weights continue to update causally during confirmation.

the confirmation sample contains **443 one-session forecasts**. the primary endpoint is adaptive versus uniform one-session normalized shape loss. the paired mean difference is approximately **−0.0000321**, with a 95% stationary-bootstrap interval of approximately **[−0.000070, 0.000004]**. the interval includes zero. five- and 22-session secondary intervals also include zero. these comparisons do not establish a directional confirmation improvement.

| one-session model | euclidean ratio | frobenius ratio | qlike ratio | gmvp annual risk (%) | gmvp+ annual risk (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| har-drd | 1.0000 | 1.0000 | 1.0000 | 9.3258 | 9.4351 |
| ghar(gl,l) | 1.0113 | 1.0109 | 1.1609 | 30.1616 | 9.7468 |
| har-cholesky | 1.0137 | 1.0062 | 1.0764 | 9.1698 | 9.4047 |
| uniform shape | 0.9982 | 0.9987 | 0.9926 | 9.2212 | 9.4664 |
| adaptive shape | 0.9977 | 0.9979 | 0.9918 | 9.2279 | 9.4912 |

ratios use har-drd as the denominator. lower is better. annualized risk uses 252 and excludes transaction costs. gmvp permits short positions; gmvp+ is long-only.

the adaptive mixture has the lowest one-session raw matrix losses among the seven tested specifications, including common-trace controls. its euclidean and frobenius mcs sets retain other models, including har-drd. it is the only model retained in the one-session qlike mcs at 5% in this candidate set. that is a secondary loss result and does not override the primary shape-loss interval.

the adaptive portfolio has less unrestricted realized risk than har-drd but more than the uniform mixture and har-cholesky. its long-only risk exceeds har-drd and the uniform mixture. learning the mixture weights therefore does not establish a portfolio gain in confirmation.

## what changed after development

development favored the raw graph model over har-drd on short-horizon matrix losses. that ordering reverses for one-session confirmation losses. a fixed uniform combination already captures most of the mixture's descriptive advantage. evaluating only the development period would give an overly favorable impression of the incremental value of adaptive weights.

the full 2022–2026 one-session adaptive cumulative shape loss is 24.6085, versus 24.6881 for the best fixed shape expert. regret is −0.0796, below the proved upper bound 2.1972. this is a comparison over the complete learning run. it is not an independently reset confirmation guarantee, nor a test of statistical significance.

the general delayed bound is conservative. at 22 sessions its value is about 2,781.76 for 1,045 origins, exceeding the trivial bound of one loss unit per origin. it therefore provides no useful numerical improvement at that horizon in this run. the sharper constant bound applies to one-session immediate feedback.

the graph portfolio's large unrestricted risk is reported rather than omitted. its maximum gross leverage is **37.05** on april 11, 2025, and its largest forecast condition number is approximately **127,174**. the adaptive mixture's corresponding maxima are **2.32** and **223.37**. these diagnostics expose inverse sensitivity and extreme positions; they do not isolate the cause of every realized loss. long-only constraints remove short positions but still do not give the graph model a confirmation risk advantage.

## use cases

- **forecast pooling:** combine distinct covariance estimates while preserving positive definiteness; retain the uniform mixture as a simple control for the value of learning.
- **risk-model review:** distinguish lower covariance error from lower allocation risk using a reproducible counterexample, raw matrix losses, and both portfolio constraint sets.
- **pair-network design:** compute stock-pair message aggregation through endpoint sums and quantify how repeated aggregation reduces centered contrasts.
- **delayed targets:** release multi-session labels only after the final retained session, preventing unresolved targets from influencing earlier mixture weights.
- **research replication:** compare the same method across development and confirmation periods and report when the paper's model ranking fails to carry over.

these are research uses. this experiment does not validate a live allocation policy or include execution costs.

## scope and reproducibility

the data are pinned public syncster minute trade bars for the manuscript's 27 stocks and spy. unit-trace covariance shapes require synchronized multivariate intraday returns, which daily univariate volatility series cannot supply. the original study's licensed lobster quote panel is unavailable here; the loader also supports locally prepared licensed minute inputs. the feed, years, price construction and retained calendar differ from the original experiment.

the strict shared-calendar rule removes sessions, particularly before 2022, so a model horizon counts retained sessions. a 1,000-observation calibration can span more than four exchange years. [replication audit](protocol-audit.md) and [paper comparison](result-comparison.md) document the implications.

run both notebooks from the covariance_replication folder with its requirements. the extension saves primary shape comparisons, all horizon-specific matrix/mcs tables, portfolio results, cumulative regret and conditioning diagnostics in docs/tables. stationary-bootstrap draws are seeded, with 10,000 replications; the primary interval's mean block length is max(5,h), and mcs uses max(2,h). all six shape intervals are pointwise, without a simultaneous multiple-testing guarantee.
