# replication notes

[project overview](../README.md) · [executed notebook](../GNNHAR_Replication.ipynb)

## data and scope

| input | what it is used for |
| --- | --- |
| [public lobster samples](https://php.lobsterdata.com/info/DataSamples.php) | verify quote alignment and the five-minute variance calculation on one session for apple, intel and microsoft |
| [oxford-man archive](https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip) | run the 120-month backtest with four u.s. equity indices, using five-minute subsampled realized variance and open-to-close returns |
| [clements–perera stock panel](https://github.com/AdamEClements/Wavelet-HAR-Reconciliation-data/tree/6e7a69a8548f9a24595f7007fb0fbbf0424b8dc3) | run the same design on 30 individual u.s. stock series with five-minute realized variance and daily returns |
| [licensed lobster data](https://lobsterdata.com/) | run the same design on the original 27 dow stocks once the historical inputs are available |

the index panel preserves the two inputs the method needs: measured intraday variance for the target and a daily return series for the graph. index overlap and the shorter optimizer budget can affect the public model rankings. the one-session lobster sample is a separate measurement check.

the notebook defaults to `SOURCE = "clements"` for the public stock panel. the notebook matches numbered stock columns across both csvs and restricts the data to the study dates. this alternative is based on refinitiv prices and has no verified mapping to the paper's 27 tickers. set `SOURCE = "oxford"` to rerun the earlier four-index benchmark.

## evaluation

models are refit monthly with up to 1,000 past sessions. the notebook keeps 250 sessions for neural validation and drops any forward label that crosses a calibration boundary. horizons include 1, 5 and 22 observed sessions. the public runs use two seeds with a 16-epoch cap. the licensed mode defaults to ten seeds and a 5,000-epoch ceiling with early stopping.

relative loss divides each model's out-of-sample loss by the corresponding har-mse loss. values below 1 favor the candidate. qlike is sensitive to very small forecasts, so the notebook flags an example where a single index-date accounts for about 5% of the one-session baseline qlike.

## licensed lobster run

place the daily panels in `data/lobster/` and switch the notebook setting to `SOURCE = "lobster"`.

| filename | contents |
| --- | --- |
| `DJIA_var_FH1.csv` | daily realized variance in percent squared for all 27 tickers |
| `DJIA_ret_FH1.csv` | daily log returns in percent for the same tickers |
| `SPY_var_FH1.csv` | optional daily spy realized variance for the stress-regime comparison |

each file starts with a session-date column. the notebook also shows how to build the first two panels from licensed one-minute best quotes. historical licensed inputs stay in the local data folder.

## read the outputs

- for daily risk monitoring, compare the graph forecast with the same model after switching off the peer branch.
- for weekly stress planning, check whether the five-session graph forecast improves on har for the loss you care about.
- for monthly planning, compare the 22-session losses before adding graph complexity. the models forecast marginal variance for individual assets. portfolio variance also requires covariance estimates.
