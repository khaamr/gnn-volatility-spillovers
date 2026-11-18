# saved covariance figures

[open the live interactive gallery](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html) · [animated graph explanation](https://khaamr.github.io/graph-covariance-forecasting/previews/graph-explainer.html) · [machine-readable inventory](index.json)

the nine figures are extracted from saved plotly mime outputs in the three executed covariance notebooks. the export does not fit models, fetch data or change the experimental results. each json payload retains its original data, layout and configuration. index.json records its source notebook, notebook hash, cell and output position.

| group | files | contents |
| --- | --- | --- |
| replication | replication-01.json to replication-03.json | horizon loss ratios, portfolio risk/turnover and a forecast-error covariance heatmap |
| shape extension | shape-extension-01.json to shape-extension-03.json | line-graph contraction, causal expert weights and one-session loss ratios |
| spectral extension | spectral-extension-01.json to spectral-extension-03.json | regime-gate weights, model loss comparison and long-only portfolio risk |

open the [live dashboard](https://khaamr.github.io/graph-covariance-forecasting/research_dashboard.html) for interactive charts. GitHub’s file viewer shows html source rather than running the dashboard. for offline use, open research_dashboard.html in a browser after downloading the project. plotly is bundled inside the html, so the charts work offline. source-notebook and json links work when the project folder is kept together.

from the repository root, regenerate the gallery with:

```sh
python covariance_replication/export_figures.py
```

from the covariance_replication folder, use python export_figures.py. install that folder's requirements first. if notebook outputs change, rerun the export to refresh the figure inventory.

static research diagrams and png figures are under covariance_replication/docs. editable svg versions are included alongside the architecture images. the blue project banner under docs/images is a labeled schematic rather than a measured covariance matrix.

2025–2026 extension figures use a reused retrospective benchmark. lower mean loss alone does not establish lower future portfolio risk. see the notebooks and paired-comparison tables for uncertainty and limitations.

## animated graph explanation

[open the live animation](https://khaamr.github.io/graph-covariance-forecasting/previews/graph-explainer.html) · [local animation file](../previews/graph-explainer.html)

scroll through five stages: individual stocks, graph connections, neighbor aggregation, the stock-pair line graph and a covariance matrix. select a stock, adjust its signal and follow the moving messages to see how neighbor context changes. playback and pulse controls are included.

this css, javascript and svg illustration uses six example stocks and synthetic values. it is an educational companion to the nine empirical plotly figures, and does not add an experimental result to their inventory. open previews/graph-explainer.html in a browser to use it locally.
