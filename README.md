# Analysis

このディレクトリは、Android アプリの拍ごとの記録と CNAP の連続血圧を同期し、評価まで回すための解析系コードをまとめたものです。  
特に入口になるのは [`run_realtime_session.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_session.py) です。

## 役割

- `run_realtime_session.py`
  - Android アプリと CNAP の同時計測を開始
  - ターミナルへ `RTBP / SinBP_D / SinBP_M / CNAP` をリアルタイム表示
  - `Ctrl+C` 停止後にスマホ CSV と CNAP beats CSV を回収
  - タイムスタンプ同期した `merged.csv` を生成
  - 誤差集計、時系列図、Markdown レポートを自動生成

- `realtime_pipeline/`
  - `android_bridge.py`: `adb` 接続確認、intent 起動、logcat 監視、CSV pull
  - `cnap_bridge.py`: CNAP 接続確認、既存 `realtime_capture.py` 起動、beats CSV 管理
  - `merge_session.py`: スマホ CSV と CNAP beats CSV の同期マージ
  - `evaluate_session.py`: 評価 CSV/JSON、図、レポート生成

- `run_realtime_coefficient_pipeline.py`
  - `Data/realtime_sessions/` 配下の全セッションから固定係数候補を学習
  - 現行アプリ係数、全データ再学習、Leave-One-Session-Out を比較評価
  - `Data/realtime_coefficient/` に係数、評価表、グラフ、予測CSVを出力

- `BP_Analysis/`
  - 現行の係数学習パイプライン本体のみを配置
  - 現在は `fit_realtime_map_pp_coefficients.py` を `run_realtime_coefficient_pipeline.py` から呼び出す

## `run_realtime_session.py`

### 何をするか

`run_realtime_session.py` はリアルタイム計測の親パイプラインです。  
1 回の実行で、計測開始から停止後の統合解析までを担当します。

処理の流れ:

1. 古い `run_realtime_session.py` / `CNAP/realtime_capture.py` を停止
2. phone と CNAP の接続状態をチェック
3. `subject_id / session_number / session_id` を確定
4. Android アプリを `adb` intent で自動開始
5. CNAP 取得を開始
6. ターミナルに同期済みライブ表示を流す
7. `Ctrl+C` 後に保存待機、スマホ CSV pull、CNAP beats 確定
8. `merged.csv`、評価 CSV/JSON、プロット、レポート生成

### 実行方法

`RealTime-IBI&BP` 直下から:

```bash
python3 Analysis/run_realtime_session.py --subject-id Nakazawa --session-number 1 --mode 1
```

`Analysis` ディレクトリへ入っている場合:

```bash
python3 run_realtime_session.py --subject-id Nakazawa --session-number 1 --mode 1
```

### 引数

- `--subject-id`
  - 被験者 ID。未指定なら対話入力

- `--session-number`
  - セッション番号。未指定なら `1`

- `--mode`
  - Android 側へ渡す mode。通常は `1`

- `--session-id`
  - 明示指定しない場合は `subject_id_YYYYMMDD_HHMMSS`

### 実行前の前提

- Android 端末が `adb devices -l` で `device` になっている
- 最新版 APK が端末に入っている
- CNAP の TUSB-ADAPIO が認識されている
- 端末はロック解除済み
- CNAP とスマホは、計測可能な状態まで事前に準備済み

### 実行中の表示

実行中は概ね次のような表示になります。

```text
[sync ] t_phone= 37.59s beat=006 CNAP= 115.4/ 89.5 RTBP= 109.8/ 79.0 SinD= 101.6/ 77.5 SinM= 111.6/ 88.3 dt=+0.12s
```

意味:

- `t_phone`
  - スマホ側の経過時間

- `beat`
  - スマホ側の拍番号

- `CNAP`
  - 近傍同期された CNAP の SBP/DBP

- `RTBP / SinD / SinM`
  - Android アプリ 3 手法の SBP/DBP

- `dt`
  - スマホ拍と CNAP 拍の時間差

### 停止方法

終了したいタイミングで `Ctrl+C` を押します。  
停止後、自動で CSV 回収と解析が走ります。

## 出力先

1 セッション分の成果物は次にまとまります。

- ルート
  - [`Analysis/Data/realtime_sessions`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions)

- 例
  - [`Analysis/Data/realtime_sessions/<session_id>`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions)

中身:

- `smartphone/`
  - Android から pull した CSV
  - `*_Training_Data.csv`
  - `*_RTBP.csv`
  - `*_SinBP_D.csv`
  - `*_SinBP_M.csv`
  - `*_Wave_Data.csv`
  - `*_元データ.csv`

- `<session_id>_merged.csv`
  - スマホと CNAP を同期した主解析 CSV

- `evaluation/`
  - `session_evaluation_summary.csv`
  - `session_evaluation_summary.json`
  - `sbp_timeseries.png`
  - `dbp_timeseries.png`
  - `session_report.md`

CNAP の元 beats CSV は別に:

- [`Analysis/Data/pdp/realtime_aux`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/Data/pdp/realtime_aux)

配下の `<session_id>/` に保存されます。

## 重要な CSV

### `*_Training_Data.csv`

Android アプリが拍ごとに保存する詳細 CSV です。  
このファイルに、3 手法の特徴量、推定値、係数、各項の寄与、reject reason、raw 値、制約後値、拍品質指標まで入ります。

主な列:

- 共通
  - `session_id`
  - `beat_index`
  - `timestamp_ms`
  - `wall_time_iso`
  - `ISO`
  - `fps`

- RTBP
  - `M1_*`
  - `M1_SBP_raw`, `M1_DBP_raw`
  - `M1_IBI_input_ms`, `M1_IBI_smoothed_ms`
  - `M1_reject_reason`

- SinBP(D)
  - `M2_*`
  - `M2_SBP_raw`, `M2_DBP_raw`
  - `M2_SBP_attempt_final`, `M2_DBP_attempt_final`
  - `M2_constraint_applied`, `M2_clamp_applied`
  - `M2_beat_sample_count`, `M2_beat_range`, `M2_beat_std`
  - `M2_systole_ratio`, `M2_diastole_ratio`
  - `M2_reject_reason`

- SinBP(M)
  - `M3_*`
  - `M3_sinPhi`, `M3_cosPhi`
  - `M3_fit_a`, `M3_fit_b`, `M3_fit_rmse`
  - `M3_SBP_raw`, `M3_DBP_raw`
  - `M3_SBP_attempt_final`, `M3_DBP_attempt_final`
  - `M3_constraint_applied`, `M3_clamp_applied`
  - `M3_reject_reason`

### `<session_id>_merged.csv`

主解析用 CSV です。  
`Training_Data.csv` の全列に CNAP の列が付加されたものです。

重要な列:

- `ref_SBP`, `ref_DBP`
  - CNAP 参照値

- `time_delta_ms`
  - スマホ拍と CNAP 拍の差

- `abs_time_delta_ms`
  - 時間差の絶対値

- `matched_cnap_beat_index`
  - 同期された CNAP 側の拍番号

## 改善時の見方

このパイプラインは、後からアルゴリズム改善をしやすくするために debug 列を多めに残しています。

見ると良い例:

- `SinBP(M)` が高すぎる
  - `M3_SBP_term_*`
  - `M3_sinPhi`, `M3_cosPhi`
  - `M3_fit_a`, `M3_fit_b`
  - `M3_fit_rmse`
  - `M3_reject_reason`

- `SinBP(D)` が不安定
  - `M2_E`
  - `M2_SBP_term_*`
  - `M2_beat_range`, `M2_beat_std`
  - `M2_systole_ratio`, `M2_diastole_ratio`

- `RTBP` が急に外れる
  - `M1_IBI_input_ms`
  - `M1_IBI_smoothed_ms`
  - `M1_SBP_raw`, `M1_DBP_raw`
  - `M1_SBP_term_*`

- 同期ズレが怪しい
  - `time_delta_ms`
  - `abs_time_delta_ms`
  - `matched_cnap_beat_index`

## よくある失敗

### `phone check failed`

- `adb devices -l` が `device` か確認
- 端末をアンロック
- USB デバッグ許可を再確認

### `cnap check failed`

- TUSB-ADAPIO の USB 認識を確認
- 前回の `realtime_capture.py` が残っていないか確認

### `smartphone training csv was not pulled`

- Android 側で停止 intent が届いたか確認
- `Download/` と `Download/PC_Sync/...` に保存されているか確認

### `CNAP=---/---`

- CNAP 側プロセスが出力を始めているか確認
- 端末側だけ進んでいないか確認

## 関連ファイル

- [`run_realtime_session.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_session.py)
- [`run_realtime_coefficient_pipeline.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/run_realtime_coefficient_pipeline.py)
- [`BP_Analysis/fit_realtime_map_pp_coefficients.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/BP_Analysis/fit_realtime_map_pp_coefficients.py)
- [`realtime_pipeline/README.md`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/README.md)
- [`realtime_pipeline/android_bridge.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/android_bridge.py)
- [`realtime_pipeline/cnap_bridge.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/cnap_bridge.py)
- [`realtime_pipeline/merge_session.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/merge_session.py)
- [`realtime_pipeline/evaluate_session.py`](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/realtime_pipeline/evaluate_session.py)
