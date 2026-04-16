# Realtime Pipeline

`run_realtime_session.py` は、CNAP と Android アプリを同時に開始し、停止後に以下を自動で実行します。

- Android の `RealtimeSession` ログから 3 手法の血圧をターミナル表示
- CNAP の beats CSV を保存
- Android の `*_Training_Data.csv` などを `adb pull`
- Android と CNAP をタイムスタンプ同期して merged CSV を作成
- `RTBP / SinBP_D / SinBP_M` の SBP/DBP 誤差を集計

実行例:

```bash
python3 Analysis/run_realtime_session.py --subject-id NY --session-number 1 --mode 1

# 既存 session の evaluation だけ回し直す（default: past は除外）
python3 Analysis/run_realtime_session.py --rerun-existing-evaluations

# past 配下も含めて回し直す
python3 Analysis/run_realtime_session.py --rerun-existing-evaluations --past
```

出力先:

- `Analysis/Data/realtime_sessions/<session_id>/smartphone`
- `Analysis/Data/realtime_sessions/<session_id>/<session_id>_merged.csv`
- `Analysis/Data/realtime_sessions/<session_id>/evaluation/session_evaluation_summary.csv`

前提:

- `adb` で端末が接続済み
- Android アプリに今回の更新版 APK が入っている
- CNAP の TUADAPIO 接続が有効
