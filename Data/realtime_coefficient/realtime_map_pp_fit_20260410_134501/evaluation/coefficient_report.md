# Realtime Coefficient Report

- sessions_root: /Users/nakazawa/Desktop/Nozawa Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions
- ridge_alpha: 1.0
- filters: abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier
- CNAP usage: offline training/evaluation labels only; not used by the runtime app.

## Outputs

- session_evaluation_summary.csv
- session_evaluation_summary.json
- current_app_evaluation_summary.csv
- refit_evaluation_summary.csv
- loso_evaluation_summary.csv
- fixed_coefficients.json

## Plots

- plots/jin_20260409_182834_sbp_timeseries.png
- plots/jin_20260409_182834_dbp_timeseries.png
- plots/yu_20260409_182115_sbp_timeseries.png
- plots/yu_20260409_182115_dbp_timeseries.png
- plots/zawa_20260409_181339_sbp_timeseries.png
- plots/zawa_20260409_181339_dbp_timeseries.png
- plots/all_sbp_scatter.png
- plots/all_dbp_scatter.png
