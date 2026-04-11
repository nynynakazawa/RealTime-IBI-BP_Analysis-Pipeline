# AROB Tracking Analysis Summary

- primary_window_seconds: 20
- session_count: 6
- representative_session: zawa_20260409_181339

## SBP at 20 s

| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Direction Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| sinBP(D) | 1.460 | 1.870 | 0.009 | 0.217 | 0.493 |
| sinBP(M) | 1.464 | 1.843 | 0.068 | 0.185 | 0.676 |
| sinBP(D-EOnly) | 1.582 | 1.858 | -0.031 | 0.035 | 0.254 |
| RTBP | 1.688 | 2.169 | -0.262 | -0.152 | 0.377 |

## DBP at 20 s

| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Direction Agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| sinBP(M) | 0.768 | 0.937 | 0.325 | 0.450 | 0.526 |
| sinBP(D) | 0.874 | 1.083 | -0.047 | 0.209 | 0.536 |
| RTBP | 0.949 | 1.205 | -0.234 | -0.107 | 0.615 |
| sinBP(D-EOnly) | 0.956 | 1.260 | -0.132 | -0.107 | 0.511 |
