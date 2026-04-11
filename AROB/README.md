# AROB Tracking Analysis

AROB 論文用の追従重視解析をここに分離する。

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
python3 Analysis/run_arob_tracking_analysis.py
```

出力先:

- `Analysis/Data/arob_tracking/tracking_eval_<timestamp>/`

