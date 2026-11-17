# comparison with the quoted paper results

the quoted ratios match the supplied manuscript's tables 4 and 6. the quoted unrestricted gmvp risks match table 5. the public replication produces different rankings and significance results.

## one-session loss comparison

| metric for ghar(gl,l) | published ratio | public ratio | published mcs p-value | public mcs p-value |
| --- | ---: | ---: | ---: | ---: |
| euclidean | 0.975 | 0.9835 | 1.000 | 0.0023 |
| frobenius | 0.975 | 0.9855 | 0.768 | 0.0008 |
| qlike | 0.982 | 0.9851 | 1.000 | 0.5572 |

an adjusted mcs p-value below 0.05 means exclusion from that model confidence set. our graph model is excluded for euclidean and frobenius loss. har-cholesky has the smallest distance losses, with ratios 0.9376 and 0.9388. an independent sq mcs implementation from the authors' gnnhar repository also selected har-cholesky alone for these two metrics on the same public losses.

ghar(k,l) is likewise excluded from our one-session euclidean and frobenius confidence sets, with adjusted p-values 0.0023 and 0.0008. both graph models remain in our qlike confidence set.

## like-for-like portfolio comparison

| model | published unrestricted gmvp risk (%) | public unrestricted gmvp risk (%) | published long-only gmvp+ risk (%) | public long-only gmvp+ risk (%) |
| --- | ---: | ---: | ---: | ---: |
| har-drd | 9.918 | 9.4165 | 9.936 | 9.3326 |
| ghar(gl,l) | 9.799 | 9.4798 | 9.901 | 9.3852 |

the public graph portfolio's realized risk is slightly higher than har-drd under both constraints. the earlier chat reply paired our long-only figure with the paper's unrestricted figure. this table corrects the comparison.

## verified differences

the authors forecast during 2011–2021 using lobster quote data. the public run forecasts during july 2022–december 2024 using syncster trade bars. those are different samples and measurement processes.

our strict 27-stock intersection removes 596 of the 1,496 spy sessions that pass screening during 2016–2021. in 2019 alone, it retains 98 of 248 usable spy sessions. consequently, the first 1,000-observation calibration spans more than six years. it cannot be interpreted as the paper's approximately four exchange years. our notebook now displays annual calendar completeness and the actual date span of each calibration window. the 2022–2024 shared panel retains every screened spy session.

the supplied manuscript uses series-specific intercepts. the released ghar scripts use a common pooled intercept. a same-data sensitivity run with common intercepts yields one-session graph ratios of 0.9872, 0.9897 and 0.9851. ghar(gl,l) is still excluded from the euclidean and frobenius mcs. this specification difference alone does not recover the original conclusions.

the paper specifies monthly refits while the released code advances by 22 rows, and its portfolio script annualizes with 250 rather than the manuscript's 252. we follow the displayed manuscript choices. changing 250 to 252 multiplies all reported risks by the same factor and cannot change their ranking.

## conclusion supported by this experiment

the graph terms lower average short-horizon losses relative to har-drd in the public sample. the public experiment does not reproduce the paper's graph-model dominance across all three mcs criteria or its one-session portfolio advantage. closeness of a few rounded ratios is insufficient to establish the stronger claims.

the paper offers the diffusion of stock-specific shocks as an interpretation of horizon decay. this experiment does not identify that mechanism causally, or isolate which data difference explains the ranking change.

[paper](https://doi.org/10.1093/jjfinec/nbae026) · [released covariance code](https://github.com/chaozhang-ox/Graph-based-HAR/tree/751f8f0b400a33e2f2fd01501ffdc42a2315f013) · [independent mcs implementation](https://github.com/chaozhang-ox/GNNHAR/blob/98a85b0802bf9336bb68196013c7bfbfa92592f8/MCS.py)
