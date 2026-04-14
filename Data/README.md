# Data Layout

## Source Data

- `realtime_sessions/`
  - session ごとの元成果物
  - `merged.csv`, `evaluation/`, smartphone/CNAP 生ログの保存先
  - 今後も削除しない

## Derived Analysis

- `arob_tracking/`
  - 論文化用の再解析結果
  - `tracking_eval_YYYYMMDD_HHMMSS/` ごとに 1 run
- `realtime_coefficient/`
  - 係数学習・比較結果
  - `realtime_map_pp_fit_YYYYMMDD_HHMMSS/` ごとに 1 run

## Auxiliary

- `pdp/realtime_aux/`
  - realtime 計測中の補助 CSV

## Current Recommendation

論文の主解析としてまず参照する run:

- `arob_tracking/tracking_eval_20260414_143908`

備考:
- `centered MAE` / `delta_corr` / `detrended_corr` の 3 指標に整理済み
- 論文用の 4 系列 `RTBP / SinBP_M / SinBP_D / SinBP_D_PPShapeC` に絞っている

係数比較の主 source:

- 現時点では `realtime_sessions/` を優先
- `realtime_coefficient/` は補助的に使う

係数探索の代表 run:

- `realtime_coefficient/realtime_map_pp_fit_20260411_180923`
