# presentation content plan

this is the source outline for a future presentation. no slide deck has been generated. all figures and reported values below already exist in the project.

## visual direction

use the anonymous navy/blue project cover, white result panels and the existing architecture diagrams. keep slide titles in lowercase. use direct links to the original article and upstream implementation for citation. distinguish diagrams labeled schematic from measured results.

## slide sequence

| slide | message | source material |
| --- | --- | --- |
| 1. research question | can graph information improve covariance forecasts and portfolio outcomes? | images/project-cover.png and the project overview |
| 2. why the full matrix matters | explain variances, co-movement and minimum-variance allocation | ../covariance_replication/docs/architecture.png |
| 3. data and scope | 27 stocks; public trade bars differ from the licensed quote panel | ../covariance_replication/docs/protocol-audit.md |
| 4. reference pipeline | stock graph, stock-pair line graph, har lags and monthly historical refits | ../covariance_replication/docs/architecture.svg |
| 5. initial replication | modest short-horizon loss gains; no replicated portfolio advantage | ../covariance_replication/docs/forecast-loss.png and result-comparison.md |
| 6. the loss–risk gap | smaller matrix error can coexist with higher true portfolio risk | ../covariance_replication/docs/guarantee-map.png and theory.md |
| 7. adaptive shapes | delayed feedback, unit-trace expert mixtures and their limited guarantee | ../covariance_replication/docs/adaptive-architecture.png |
| 8. geometric spectral model | five experts, regime gate, eigenvalue cleaning and equal objectives | ../covariance_replication/docs/spectral-architecture.png |
| 9. accuracy–risk trade-off | show every benchmark control; compare the static and neural pools | ../figures/spectral-extension-02.json and spectral-extension-03.json |
| 10. uncertainty and next steps | risk interval includes zero; new confirmation data are needed | ../covariance_replication/docs/spectral/tables/paired-comparisons.csv |

## figures to prioritize

- the original pipeline architecture for the method section.
- replication loss ratios for the short-horizon effect.
- the geometric spectral architecture for the extension.
- the full benchmark loss and long-only risk comparisons together.
- causal weights only if there is time to explain what information is available at each origin.

all interactive figures are available in ../research_dashboard.html. static architecture pngs are suitable for immediate slide use; the svg sources support editing.

## exact headline values

on the same 443 reused benchmark origins, the geometric spectral gate has frobenius ratio 0.9821 against har-drd, approximately 1.79% lower mean distance. long-only annualized risk is 9.4061%, versus 9.4351% for har-drd. the paired long-only variance interval includes zero. the predefined joint criterion is not passed.

the static pool has lower matrix-distance point losses than the neural gate. arithmetic har covariance has lower portfolio risk but higher matrix losses. show this trade-off rather than selecting only favorable comparisons.

## claims to keep precise

these are public-data method replication results, rather than a numerical reproduction of the original licensed study. the proofs are conditional specializations of established ideas. the later benchmark was reused during architecture research. a mathematical conditioning bound does not guarantee lower realized portfolio risk.
