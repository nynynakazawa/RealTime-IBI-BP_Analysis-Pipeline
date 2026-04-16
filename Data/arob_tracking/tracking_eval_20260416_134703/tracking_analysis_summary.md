# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 8
- representative_session: eitaro_20260410_155023
- paper_core_methods: RTBP, SinBP_M, SinBP_D
- paper_supplemental_methods: 

## SBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 8.203 | 8.446 | 0.191 | -1.025 | 1.650 | -0.012 | 0.094 |
| sinBP(M) | 7.889 | 8.223 | -0.003 | -1.575 | 1.695 | 0.028 | -0.034 |
| RTBP | 8.803 | 9.256 | -0.129 | -3.145 | 1.943 | -0.026 | -0.154 |

## DBP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 4.564 | 4.675 | 0.184 | 0.168 | 0.937 | 0.052 | 0.125 |
| sinBP(D) | 5.080 | 5.214 | 0.038 | -0.163 | 0.983 | 0.135 | 0.109 |
| RTBP | 6.119 | 6.318 | -0.108 | -1.524 | 1.047 | 0.295 | 0.046 |

## PP at 20 s

| Method | Mean MAE | Mean RMSE | Mean Corr | Mean Bias | Mean cMAE | Mean dCorr | Mean hpCorr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 11.575 | 12.003 | 0.097 | -1.744 | 2.449 | 0.009 | 0.052 |
| sinBP(D) | 12.682 | 13.095 | 0.120 | -0.861 | 2.501 | 0.041 | 0.114 |
| RTBP | 12.679 | 13.335 | -0.072 | -1.621 | 2.888 | 0.122 | 0.002 |
