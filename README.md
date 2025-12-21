# Traditional Dance Analyzer

伝統芸能の「身体知」を可視化するAIアプリケーションです。
動画から骨格と音声を解析し、「きれ」「間」「腰の据わり」といった暗黙知の要素を定量的なデータとして提示します。

## 概要

本システムは、モダンなWeb技術とAI解析技術を組み合わせ、稽古や学習の支援を行うことを目的としています。

### 主な機能
- **高度な動画再生**: 解析された骨格データ（スケルトン）を元動画にオーバーレイ表示。フレーム単位でのコマ送りやスロー再生が可能。
- **データ可視化**: 「腰の据わり（安定性）」「リズム調和度」「序破急（ダイナミズム）」を時系列グラフで表示。
- **AIフィードバック**: 改善点や良かった点についてのAIによるテキストフィードバック。
- **LINE Login 連携**: ユーザー認証機能（モック実装済み）。

## システム構成

マイクロサービスアーキテクチャを採用し、Docker Composeによって管理されています。

| Service | Technology Stack | Description |
| --- | --- | --- |
| **Frontend** | Next.js (App Router), Tailwind CSS, Video.js, Recharts | ユーザーインターフェース。動画再生、グラフ表示、操作を担当。 |
| **Backend** | FastAPI, MediaPipe Pose, Librosa, OpenCV | 解析エンジン。動画の骨格抽出、音声リズム解析、API提供。 |
| **Database** | MySQL 8.0 | ユーザー情報、解析履歴の保存。 |
| **Vector DB** | Qdrant | (将来拡張用) 動きの類似度検索やアーカイブ検索用。 |

## ディレクトリ構造

```
.
├── backend/                # FastAPI Application
│   ├── analyzer.py         # Analysis Logic (MediaPipe/Librosa)
│   ├── main.py             # API Endpoints
│   ├── database.py         # DB Connection
│   └── Dockerfile
├── frontend/               # Next.js Application
│   ├── app/                # App Router Pages (dashboard, login)
│   ├── components/         # UI Components (VideoAnalyzer)
│   └── Dockerfile
├── docker-compose.yml      # Container Orchestration
└── README.md
```

## 動作環境 (System Requirements)

YOLOv8-PoseやPyTorchなどの大規模なライブラリを導入したことで、リソースの要求スペックは以前のMediaPipeのみの構成から大幅に上昇しています。具体的に必要となるメモリとストレージの目安は以下の通りです。

### 1. ストレージ（ディスク容量）

Dockerの割り当て容量（Disk Limit）は、**最低でも 40GB以上、推奨 64GB以上** です。

* **イメージサイズ（約 4〜6GB）**:
    * PyTorch (`torch`) + `torchvision`: CPU版でも約2GB以上を消費します。
    * その他の依存ライブラリ (`ultralytics`, `opencv`, `mediapipe`, `librosa` 等): 合計で1GB以上になります。
    * ベースとなるPythonイメージやOSパッケージ: 約1GB。

* **ビルドキャッシュ（約 10〜20GB）**:
    * Dockerはビルドを高速化するために中間レイヤーを保存します。大規模なパッケージのインストール中は、一時的にイメージ本体の2〜3倍の容量を必要とすることがあります。

* **動画・モデルの保存**:
    * 解析対象の動画ファイルや、実行時に自動ダウンロードされるYOLOの学習済みモデル（`.pt`ファイル）用のスペースも必要です。

### 2. メモリ（RAM）

Dockerへの割り当てメモリは、**最低でも 8GB以上、推奨 16GB** です。

* **ビルド・インストール時（2〜4GB消費）**:
    * `pip install` で `torch` などの巨大なホイールファイルを展開・配置する際、一時的にメモリ消費が激しくなります。

* **実行時の推論（2〜4GB以上消費）**:
    * YOLOv8-Poseモデルをメモリ上にロードし、動画の各フレームを画像として展開して解析を行うため、定常的に多くのメモリを占有します。

* **音声解析（1GB〜）**:
    * `librosa` で長尺の動画から音声をロードして解析する場合、動画の長さに比例してメモリを消費します。

* **フロントエンド + バックエンド同時実行**:
    * 両方のコンテナを同時に動かすため、OS全体として余裕を持たせる必要があります。

### 設定の目安（Docker Desktop）

もしPCの物理メモリが16GBであれば、Dockerへの割り当てを以下のように設定してみてください。

* **Memory**: 8GB 〜 12GB
* **Swap**: 2GB
* **Disk image size**: 64GB

これ以下の設定（例えば初期設定の2GBや4GB）では、今回発生したような書き込みエラー（Input/output error）や、インストール中のクラッシュが発生しやすくなります。余裕を持った設定に変更した後、一度 **`docker system prune`** を実行して古いキャッシュを掃除してから再ビルドすることをお勧めします。

## 使い方 (Getting Started)

### 前提条件
- Docker および Docker Compose がインストールされていること。

### 起動方法

1. リポジトリをクローンします。
2. プロジェクトのルートディレクトリで以下のコマンドを実行し、コンテナをビルド・起動します。

```bash
docker-compose up --build
```

初回起動時はビルドに数分かかる場合があります。

3. ブラウザで以下のURLにアクセスします。

- **フロントエンド (UI)**: [http://localhost:3000](http://localhost:3000)
- **バックエンド (API Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 操作フロー

1. **ログイン**: ログイン画面が表示されたら、「Login with LINE」ボタンをクリックします（現在はモック認証で自動的にダッシュボードへ遷移します）。
2. **動画アップロード**: ダッシュボードのファイル選択ボタンから、解析したい踊りの動画ファイル（MP4等）を選択し、「Analyze」ボタンを押します。
3. **解析と確認**:
    - 解析状況がプログレスバーで表示されます。
    - 解析完了後、左側のプレイヤーで骨格付き動画を確認できます。
    - 右側や下部のグラフで、安定性やエネルギーの変化を確認できます。
