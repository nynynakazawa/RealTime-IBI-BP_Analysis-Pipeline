# Session Report

- samples_with_reference: 282
- total_samples: 285
- evaluation_filters: abs_time_delta_ms<=350.0, output_valid=true, reject_reason=ok, artifact_flag=0

## Metrics

| method | target | filters | n | mae | rmse | corr |
| --- | --- | --- | --- | --- | --- | --- |
| RTBP | SBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 279 | 11.595557375347669 | 13.21398044451716 | -0.1955858370036124 |
| RTBP | DBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 279 | 6.170309263555556 | 8.697824978461991 | 0.1319800150652094 |
| SinBP_D | SBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 256 | 15.10801340238672 | 17.01197154525213 | -0.1212125093442909 |
| SinBP_D | DBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 256 | 11.912926174835938 | 13.954444556875462 | 0.0741244404864194 |
| SinBP_M | SBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 256 | 5.644539359277345 | 7.559804956044351 | -0.1546335671328617 |
| SinBP_M | DBP | abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0 | 256 | 5.205023589664062 | 6.6830194961150005 | 0.0888338245619455 |

## Plots

- sbp_timeseries.png
- dbp_timeseries.png

