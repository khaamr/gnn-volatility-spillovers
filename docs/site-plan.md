# site content plan

this note organizes assets for a future current-project website. no live site has been enabled for this repository. the standalone research_dashboard.html is an offline figure artifact and can later be reused as an interactive results page.

## proposed navigation

| page | purpose | existing assets |
| --- | --- | --- |
| overview | state the question, scope and supported findings | ../README.md and images/project-cover.png |
| method | show data construction, stock/pair graphs and model families | ../covariance_replication/docs/architecture.png and protocol-audit.md |
| replication | compare the public results with the original study | ../covariance_replication/docs/result-comparison.md |
| extensions | explain adaptive shapes and geometric spectral cleaning | ../covariance_replication/docs/adaptive-architecture.png and spectral-architecture.png |
| results | place matrix error and portfolio risk side by side | ../research_dashboard.html and ../figures/index.json |
| theory | present precise statements with assumptions and full proofs | ../covariance_replication/docs/theory.md |
| reproduce | link notebooks, requirements, input provenance and audits | ../covariance_replication/README.md |
| presentation | add the future slide deck once it exists | presentation-plan.md |

## reusable content

figures/index.json supplies source labels, notebook hashes and figure paths. figure json files can be loaded with plotly without rerunning models. the existing standalone html bundles plotly for offline use; a future site may use the same plots with an explicitly managed javascript dependency.

result tables live in covariance_replication/docs/tables and covariance_replication/docs/spectral/tables. preserve model names, comparison periods, units and confidence intervals. use the provided png architecture images initially and the svg versions for later visual editing.

## display requirements

- keep the public-data scope visible beside headline results.
- label the 2025–2026 period as a reused retrospective benchmark.
- display the static control and other strong baselines alongside the neural model.
- distinguish annualized standard deviation from variance changes and matrix-loss ratios.
- make the uncertainty explanation visible without requiring a hover.
- retain a direct path to the underlying notebooks, tables and original article.

## future publishing step

once the presentation and design are ready, select the hosting approach, review the complete pages and publish. the current repository topics and overview are already suitable for discovery; its homepage field remains available for the future site url.
