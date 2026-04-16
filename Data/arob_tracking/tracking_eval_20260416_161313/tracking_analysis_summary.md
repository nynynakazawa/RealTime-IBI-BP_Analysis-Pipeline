# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 1
- representative_session: jin_20260416_160710
- paper_core_methods: RTBP, SinBP_M, SinBP_D
- paper_supplemental_methods: 

## SBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 3.672 | 3.817 | -0.117 | -3.672 | 0.868 | 0.218 | 0.072 |
| sinBP(M) | 3.375 | 3.559 | -0.201 | -3.375 | 0.888 | -0.339 | -0.277 |
| RTBP | 3.394 | 3.561 | -0.219 | -3.394 | 0.907 | 0.247 | 0.138 |

## DBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 1.820 | 1.891 | -0.111 | 1.820 | 0.431 | 0.250 | 0.087 |
| sinBP(M) | 1.684 | 1.773 | -0.174 | 1.684 | 0.438 | -0.302 | -0.251 |
| RTBP | 1.689 | 1.772 | -0.223 | 1.689 | 0.457 | 0.272 | 0.151 |

## PP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 5.492 | 5.708 | -0.115 | -5.492 | 1.300 | 0.228 | 0.077 |
| sinBP(M) | 5.059 | 5.332 | -0.192 | -5.059 | 1.325 | -0.327 | -0.268 |
| RTBP | 5.082 | 5.333 | -0.220 | -5.082 | 1.364 | 0.255 | 0.142 |
