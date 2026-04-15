# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 8
- representative_session: eitaro_20260410_155023
- paper_core_methods: RTBP, SinBP_M, SinBP_D
- paper_supplemental_methods: SinBP_D_PPShapeC

## SBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D-PP-C) | 4.046 | 4.449 | 0.402 | 2.209 | 1.514 | 0.847 | 0.697 |
| sinBP(D) | 4.013 | 4.391 | 0.425 | 2.114 | 1.532 | 0.767 | 0.640 |
| RTBP | 5.496 | 5.790 | 0.578 | -0.331 | 1.565 | 0.821 | 0.711 |
| sinBP(M) | 4.559 | 4.934 | 0.251 | 0.161 | 1.570 | 0.813 | 0.577 |

## DBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D-PP-C) | 2.005 | 2.206 | 0.385 | -1.099 | 0.751 | 0.844 | 0.676 |
| sinBP(D) | 1.989 | 2.179 | 0.413 | -1.049 | 0.763 | 0.753 | 0.627 |
| RTBP | 2.730 | 2.877 | 0.582 | 0.162 | 0.780 | 0.819 | 0.710 |
| sinBP(M) | 2.278 | 2.463 | 0.251 | -0.092 | 0.781 | 0.805 | 0.578 |

## PP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D-PP-C) | 6.051 | 6.655 | 0.397 | 3.308 | 2.264 | 0.846 | 0.690 |
| sinBP(D) | 6.002 | 6.570 | 0.421 | 3.163 | 2.295 | 0.763 | 0.636 |
| RTBP | 8.226 | 8.667 | 0.580 | -0.493 | 2.344 | 0.820 | 0.711 |
| sinBP(M) | 6.838 | 7.397 | 0.251 | 0.253 | 2.351 | 0.811 | 0.578 |
