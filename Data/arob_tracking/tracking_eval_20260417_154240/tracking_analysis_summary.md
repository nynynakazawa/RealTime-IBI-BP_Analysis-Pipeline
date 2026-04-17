# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 6
- representative_session: eitaro_20260416_165247
- methods: RTBP, SinBP_M, SinBP_D

## SBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 6.267 | 6.545 | 0.254 | -6.267 | 1.186 | 0.033 | 0.175 |
| RTBP | 6.328 | 6.575 | -0.016 | -6.234 | 1.212 | -0.142 | 0.020 |
| sinBP(D) | 6.546 | 6.864 | 0.175 | -6.532 | 1.343 | 0.081 | 0.163 |

## DBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 3.082 | 3.223 | 0.258 | 3.082 | 0.588 | 0.040 | 0.176 |
| RTBP | 3.115 | 3.240 | -0.019 | 3.065 | 0.603 | -0.138 | 0.017 |
| sinBP(D) | 3.222 | 3.378 | 0.177 | 3.209 | 0.668 | 0.095 | 0.167 |

## PP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 9.350 | 9.768 | 0.256 | -9.350 | 1.773 | 0.035 | 0.175 |
| RTBP | 9.443 | 9.815 | -0.017 | -9.299 | 1.815 | -0.140 | 0.019 |
| sinBP(D) | 9.768 | 10.242 | 0.175 | -9.741 | 2.011 | 0.085 | 0.164 |
