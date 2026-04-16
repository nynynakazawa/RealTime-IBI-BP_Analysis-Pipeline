# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 9
- representative_session: eitaro_20260410_155023
- paper_core_methods: RTBP, SinBP_M, SinBP_D
- paper_supplemental_methods: 

## SBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 5.590 | 5.921 | 0.262 | -0.958 | 1.508 | -0.039 | 0.128 |
| RTBP | 5.716 | 6.164 | 0.170 | -0.816 | 1.588 | -0.129 | 0.028 |
| sinBP(M) | 5.716 | 6.127 | -0.056 | -0.860 | 1.663 | 0.040 | -0.058 |

## DBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 2.785 | 2.949 | 0.259 | 0.464 | 0.749 | -0.037 | 0.119 |
| RTBP | 2.848 | 3.068 | 0.168 | 0.394 | 0.792 | -0.126 | 0.026 |
| sinBP(M) | 2.851 | 3.053 | -0.056 | 0.419 | 0.831 | 0.042 | -0.062 |

## PP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 8.375 | 8.870 | 0.261 | -1.422 | 2.256 | -0.039 | 0.125 |
| RTBP | 8.564 | 9.231 | 0.170 | -1.210 | 2.379 | -0.128 | 0.027 |
| sinBP(M) | 8.567 | 9.180 | -0.056 | -1.279 | 2.494 | 0.040 | -0.059 |
