# BP_Analysis

現行の係数学習パイプライン本体だけを置くディレクトリです。

入口は `../run_realtime_coefficient_pipeline.py` で、ここから
`fit_realtime_map_pp_coefficients.py` を呼び出します。

## 残しているファイル

- `fit_realtime_map_pp_coefficients.py`
  - default では `Analysis/Data/realtime_sessions/` 配下（`past/` 除外）の全セッションを読み込む
  - `--past` を付けると `Analysis/Data/realtime_sessions/past/` も含める
  - 現行アプリ係数、全データ再学習、Leave-One-Session-Out を評価する
  - `Analysis/Data/realtime_coefficient/` に係数、評価CSV/JSON、グラフ、予測CSVを出力する

## 実行例

```bash
# 現行データのみ（default）
python3 Analysis/run_realtime_coefficient_pipeline.py

# 過去データも含める
python3 Analysis/run_realtime_coefficient_pipeline.py --past
```

旧 `train_bp_models.py`、`run_full_pipeline.py`、探索用スクリプト、旧結果ディレクトリは、現行のリアルタイムセッション/係数パイプラインから参照されないため削除済みです。
