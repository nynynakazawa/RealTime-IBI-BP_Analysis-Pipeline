# Current State

最終更新: 2026-04-16

この文書の定数・代表 run は [current_direction.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/current_direction.py) を正本とする。

## 1. 目的

- 主題は `within-session BP tracking`（セッション内の上下追従）
- 正本データは [Analysis/Data/realtime_sessions](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions)
- 論文主比較は `RTBP / SinBP_M / SinBP_D` の 3 手法

## 2. ディレクトリ責務

- 実運用パイプライン
  - [run_realtime_session.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_session.py)
  - [realtime_pipeline](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline)
- 論文化再解析
  - [AROB](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/AROB)
  - [run_arob_tracking_analysis.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_arob_tracking_analysis.py)
- 係数探索
  - [BP_Analysis](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/BP_Analysis)
  - [run_realtime_coefficient_pipeline.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_coefficient_pipeline.py)

## 3. AROB スクリプト整理

- AROB 実行エントリポイントは 1 本:
  - [run_arob_tracking_analysis.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_arob_tracking_analysis.py)
- 一時チューニングスクリプトは削除済み:
  - `run_arob_nonrtbp_overtake.py`
  - `run_arob_nonrtbp_probe.py`
  - `run_arob_ppshapec_fineprobe.py`
  - `run_arob_sind_sinm_boost.py`
  - `run_arob_tracking_autotune.py`

## 4. 手法定義の現状

- live app:
  - `RTBP / SinBP_D / SinBP_M`
- AROB 比較:
  - `RTBP / SinBP_D / SinBP_M`
- `SinBP_D_PPShapeC` は現行の実行系・比較系から除外済み

## 5. 現在の代表 run

- AROB tracking:
  - [tracking_eval_20260416_103134](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/arob_tracking/tracking_eval_20260416_103134)
- realtime coefficient:
  - [realtime_map_pp_fit_20260416_102539](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_coefficient/realtime_map_pp_fit_20260416_102539)

## 6. 保持/削除ルール

- 保持:
  - `Data/realtime_sessions/`
  - `Data/pdp/realtime_aux/`
- 削除対象:
  - 一時チューニングスクリプト
  - `__pycache__/`
  - 一時生成の検証ファイル（再生成可能なもの）
