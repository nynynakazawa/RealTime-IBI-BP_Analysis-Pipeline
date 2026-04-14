# Current State

最終更新: 2026-04-14

この文書の内容は、コード側では [current_direction.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/current_direction.py) を単一ソースとしている。

## 1. 現在の方針

- 主題は `absolute cross-subject BP estimation` ではなく `within-session BP tracking`
- 正本データは [Analysis/Data/realtime_sessions](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions)
- app の live 実装と、Analysis 側の論文化用再解析は分けて扱う
- 論文の主比較は `RTBP / SinBP_M / SinBP_D`
- 論文の補足比較は `SinBP_D_PPShapeC` のみ残す

## 2. 正本と代表 run

正本:

- [Analysis/Data/realtime_sessions](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions)
- [Analysis/Data/pdp/realtime_aux](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/pdp/realtime_aux)

現在の代表 run:

- AROB tracking:
  - [tracking_eval_20260414_143908](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/arob_tracking/tracking_eval_20260414_143908)
- realtime coefficient search:
  - [realtime_map_pp_fit_20260411_180923](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_coefficient/realtime_map_pp_fit_20260411_180923)

この 2 つの run の場所は [analysis_manifest.json](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/analysis_manifest.json) にも記録している。

## 3. Android app の現在地

入口:

- [MainActivity.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/MainActivity.java)
- [GreenValueAnalyzer.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/GreenValueAnalyzer.java)

live 血圧推定本体:

- [RealtimeBP.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/RealtimeBP.java)
  - RTBP
- [SinBPDistortion.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/SinBPDistortion.java)
  - SinBP_D
- [SinBPModel.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/SinBPModel.java)
  - SinBP_M
- [RealtimeMapPpModels.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/bp/RealtimeMapPpModels.java)
  - live app 用の MAP/PP-first 線形係数
- [BPPostProcessor.java](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/RealTime-IBI-BP/app/src/main/java/com/nakazawa/realtimeibibp/BPPostProcessor.java)
  - smoothing / calibration の共通後段

app に現在入っているもの:

- live 表示系列は `RTBP / SinBP_D / SinBP_M`
- 3 手法とも `MAP/PP` を先に作ってから `SBP/DBP` を再構成
- postprocess 後の値を表示・保存
- `Training_Data.csv` に raw / smoothed / calibrated と係数寄与項を保存
- `SinBP_D` の比較枝 `EOnly / E2 / LocalA` も CSV に保存

app に現在入っていないもの:

- `SinBP_D_PPShapeC`
  - これは Analysis 側の論文化用再解析専用
  - live app runtime には未反映

## 4. リアルタイム計測パイプライン

入口:

- [run_realtime_session.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_session.py)

内部モジュール:

- [android_bridge.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/android_bridge.py)
- [cnap_bridge.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/cnap_bridge.py)
- [merge_session.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/merge_session.py)
- [evaluate_session.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/evaluate_session.py)
- [experimental_repair.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/experimental_repair.py)
- [map_pp_runtime.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/map_pp_runtime.py)

できること:

- phone / CNAP の接続確認
- app session 開始 / 停止
- CNAP beats CSV の保存
- app Training CSV の pull
- merged CSV の生成
- session evaluation の再実行
- 古い experimental summary の repair

出力先:

- `Analysis/Data/realtime_sessions/<session_id>/`

## 5. 再学習・係数探索

入口:

- [run_realtime_coefficient_pipeline.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_coefficient_pipeline.py)

本体:

- [fit_realtime_map_pp_coefficients.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/BP_Analysis/fit_realtime_map_pp_coefficients.py)

用途:

- `realtime_sessions` をまとめて読む
- current app / refit / LOSO を比較する
- fixed coefficient 候補を作る

位置づけ:

- これは探索系
- app 反映前の候補生成に使う
- 論文の主結果には使わず、必ず `AROB` 側で再評価してから採否を決める

## 6. AROB 用の論文化解析

入口:

- [run_arob_tracking_analysis.py](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_arob_tracking_analysis.py)

本体:

- [Analysis/AROB](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/AROB)

役割:

- `realtime_sessions` を read-only に再利用
- 20 s window を主条件として tracking を比較
- 指標は `centered MAE / delta_corr / detrended_corr`
- 論文用の boxplot と被験者別 timeseries を出す

この系で残している系列:

- core:
  - `RTBP`
  - `SinBP_M`
  - `SinBP_D`
- supplemental:
  - `SinBP_D_PPShapeC`

diagnostic 系列:

- `SinBP_D_EOnly`
- `SinBP_D_E2`
- `SinBP_D_LocalA`
- `SinBP_D_PPShapeA`
- `SinBP_D_PPShapeB`

これらは `full` 出力や診断には残るが、論文の主比較には出さない。

## 7. 現在の評価上の結論

既存 6 session の再解析では:

- `SinBP_D_PPShapeC`
  - `centered MAE` と `delta_corr` が最良
- `RTBP`
  - `detrended_corr` が最良
- `SinBP_D`
  - app 本体の core 系列としては `SinBP_M` より強い
- `SinBP_M`
  - 既存データでは tracking 指標が弱い

したがって現在の整理はこうする。

- live app の本線:
  - `RTBP / SinBP_D / SinBP_M`
- 論文化の主比較:
  - `RTBP / SinBP_M / SinBP_D`
- 論文化の補足比較:
  - `SinBP_D_PPShapeC`

## 8. 消さないもの

- [Analysis/Data/realtime_sessions](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions)
- [Analysis/Data/pdp/realtime_aux](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/pdp/realtime_aux)
- [Analysis/Data/realtime_coefficient/realtime_map_pp_fit_20260411_180923](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_coefficient/realtime_map_pp_fit_20260411_180923)
- [Analysis/Data/arob_tracking/tracking_eval_20260414_143908](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/arob_tracking/tracking_eval_20260414_143908)

理由:

- raw または current representative run だから

## 9. 消してよいもの

- `__pycache__/`
- LaTeX の補助生成物
- 一時 probe ファイル

このファイルを読めば、今どこが本番系で、どこが論文用再解析で、どこが探索系か分かる状態を維持する。
