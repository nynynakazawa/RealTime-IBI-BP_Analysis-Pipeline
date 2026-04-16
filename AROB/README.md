# AROB Tracking Analysis

AROB 論文用の追従重視解析をここに分離する。

現在の正本データ・主比較系列・代表 run は
`../CURRENT_STATE.md` と `../analysis_manifest.json` を参照する。

責務:

- `io.py`
  - realtime session の読み込みと long format 化
- `windowing.py`
  - non-overlapping window 平均化
- `metrics.py`
  - session-centered tracking metrics の計算
- `plots.py`
  - 論文用の図生成
- `report.py`
  - Markdown summary と metadata 出力
- `pipeline.py`
  - 全体 orchestration

実行:

```bash
# 現行データのみ（default: realtime_sessions/past は除外）
python3 Analysis/run_arob_tracking_analysis.py

# 過去データも含める
python3 Analysis/run_arob_tracking_analysis.py --past

# 例: 特定セッションのみ
python3 Analysis/run_arob_tracking_analysis.py --session-id jin_20260416_160710

# 例: 特定セッション + past も探索対象に含める
python3 Analysis/run_arob_tracking_analysis.py --session-id zawa_20260414_154020 --past

python3 Analysis/run_arob_tracking_analysis.py --no-plots
python3 Analysis/run_arob_tracking_analysis.py --output-root Analysis/Data/arob_tracking
```

出力先:

- `Analysis/Data/arob_tracking/tracking_eval_<timestamp>/`
