# Session Report

- samples_with_reference: 185
- total_samples: 188
- evaluation_filters: abs_time_delta_ms<=350.0, output_valid=true, reject_reason=ok, artifact_flag=0

## Metrics

| method | target | filters | n | mae | rmse | corr |
| --- | --- | --- | --- | --- | --- | --- |
| RTBP | SBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 174 | 6.5087044429137935 | 8.213225054327602 | 0.0336313558106742 |
| RTBP | DBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 174 | 6.187668344672414 | 9.151225128557194 | -0.0417191095633704 |
| SinBP_D | SBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 161 | 9.724400108403726 | 12.700180726240555 | 0.0907741876354156 |
| SinBP_D | DBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 161 | 12.154670395565216 | 14.57412131693274 | -0.1047148768038589 |
| SinBP_M | SBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 157 | 6.220813553719744 | 7.31366134709348 | 0.0436178887430086 |
| SinBP_M | DBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 157 | 4.271724896509554 | 5.058236810815354 | 0.0144288111870068 |

## Plots

- sbp_timeseries.png
- dbp_timeseries.png

