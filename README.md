# Analysis

現在の実装状態と推奨 run は [CURRENT_STATE.md](/Users/nakazawa/Desktop/Nozawa%20Lab/RealTime-IBI&BP/Analysis/CURRENT_STATE.md) を見る。
コード側の共有定義は `current_direction.py` を正本とする。

このディレクトリは次の 3 層に分けて扱う。

## 1. 実運用コード

- `run_realtime_session.py`
- `realtime_pipeline/`

目的:
- スマホ + CNAP の realtime 計測
- session ごとの CSV 生成
- session ごとの evaluation

この系は「今後も回す本番系」として扱う。

## 2. 論文化用解析

- `AROB/`
- `run_arob_tracking_analysis.py`

目的:
- 既存 `realtime_sessions` を再利用した追従解析
- centered / delta / detrended 指標の比較
- 論文用の図表出力

この系は「論文で何を主張するか」を整理するための read-only 再解析系として扱う。
`realtime_sessions` の元データは消さず、そこから再計算する。

## 3. 係数探索・実験系

- `BP_Analysis/`
- `run_realtime_coefficient_pipeline.py`

目的:
- fixed coefficient の探索
- MAP/PP 系の回帰候補比較

この系は「試行錯誤の実験系」として扱う。
論文の主結果に採用する前に、必ず `AROB/` で再評価する。

## Data 配下の扱い

- `Data/realtime_sessions/`
  - 元の session 単位成果物
  - 消さない
  - 論文・再解析の source of truth
- `Data/arob_tracking/`
  - 論文化用の再解析結果
  - run ごとの出力
- `Data/realtime_coefficient/`
  - 係数学習・比較の出力
  - run ごとの出力
- `Data/pdp/realtime_aux/`
  - CNAP 補助出力

## 運用ルール

1. 元データは `Data/realtime_sessions/` を正本にする。
2. 新しい主張を作るときは、まず `AROB/` で read-only に再解析する。
3. app に反映するのは、`AROB` で評価軸が固まってからにする。
4. `arob_tracking` と `realtime_coefficient` は増えてよいが、論文で使う run は manifest に明記する。
5. `Data/pdp/realtime_aux/` は realtime 計測の補助 raw なので、論文用 run ではなくても原則残す。
