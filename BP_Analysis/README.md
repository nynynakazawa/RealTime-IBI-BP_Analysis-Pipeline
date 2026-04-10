# BP_Analysis

現行の係数学習パイプライン本体だけを置くディレクトリです。

入口は `../run_realtime_coefficient_pipeline.py` で、ここから
`fit_realtime_map_pp_coefficients.py` を呼び出します。

## 残しているファイル

- `fit_realtime_map_pp_coefficients.py`
  - `Analysis/Data/realtime_sessions/` 配下の全セッションを読み込む
  - 現行アプリ係数、全データ再学習、Leave-One-Session-Out を評価する
  - `Analysis/Data/realtime_coefficient/` に係数、評価CSV/JSON、グラフ、予測CSVを出力する

旧 `train_bp_models.py`、`run_full_pipeline.py`、探索用スクリプト、旧結果ディレクトリは、現行のリアルタイムセッション/係数パイプラインから参照されないため削除済みです。
