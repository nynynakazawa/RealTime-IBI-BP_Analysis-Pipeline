# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 6
- representative_session: eitaro_20260410_155023

## SBP at 20 s

| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Amp Ratio | Mean Direction Agreement | Inversion-Like Sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 1.460 | 1.870 | 0.009 | 0.217 | 0.855 | 0.493 | 0 |
| sinBP(M) | 1.464 | 1.843 | 0.068 | 0.185 | 0.762 | 0.676 | 0 |
| sinBP(D-EOnly) | 1.582 | 1.858 | -0.031 | 0.035 | 0.817 | 0.254 | 0 |
| RTBP | 1.688 | 2.169 | -0.262 | -0.152 | 0.472 | 0.377 | 0 |

## DBP at 20 s

| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Amp Ratio | Mean Direction Agreement | Inversion-Like Sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 0.768 | 0.937 | 0.325 | 0.450 | 1.102 | 0.526 | 0 |
| sinBP(D) | 0.874 | 1.083 | -0.047 | 0.209 | 1.229 | 0.536 | 0 |
| RTBP | 0.949 | 1.205 | -0.234 | -0.107 | 0.770 | 0.615 | 0 |
| sinBP(D-EOnly) | 0.956 | 1.260 | -0.132 | -0.107 | 1.379 | 0.511 | 0 |

## MAP at 20 s

| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Amp Ratio | Mean Direction Agreement | Inversion-Like Sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTBP | 0.267 | 0.340 | -0.250 | -9.569 | 25.079 | 0.470 | 0 |
| sinBP(D) | 0.330 | 0.407 | 0.097 | -1.700 | 35.965 | 0.564 | 0 |
| sinBP(M) | 0.373 | 0.462 | 0.021 | 5.048 | 29.862 | 0.554 | 0 |
| sinBP(D-EOnly) | 0.597 | 0.707 | -0.244 | -11.220 | 56.315 | 0.479 | 0 |

## PP at 20 s

| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Amp Ratio | Mean Direction Agreement | Inversion-Like Sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 2.001 | 2.507 | 0.208 | 0.271 | 0.687 | 0.520 | 2 |
| sinBP(D) | 2.176 | 2.730 | -0.013 | 0.226 | 0.896 | 0.426 | 4 |
| sinBP(D-EOnly) | 2.303 | 2.801 | -0.126 | 0.036 | 0.753 | 0.287 | 3 |
| RTBP | 2.500 | 3.183 | -0.186 | -0.129 | 0.458 | 0.411 | 4 |
