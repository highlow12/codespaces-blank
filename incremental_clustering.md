# 증분 클러스터링 사용법

`incremental_clustering.py`는 기존 계층형 PCA64 + 구면 FCM 결과를 상태 파일로 저장하고, 이후 임베딩을 기존 클러스터에 배정한다. 신규 배치의 노이즈 비율이 `--noise-threshold`를 초과하면 누적 전체 데이터로 자동 재클러스터링한다.

AG News 검증에서는 라벨별 1,000개가 묶여 있는 원본 특성을 고려해 각 라벨에서 100개씩 신규 배치로 추출했다. 따라서 초기 3,600개와 신규 400개 모두 네 라벨을 균등하게 포함한다.

## 초기 90% 학습

```bash
python incremental_clustering.py fit \
  --input-json results_test/ag_news_embeddings_4x1000.json \
  --start 0 --limit 3600 \
  --state-output results_incremental/ag_news_90.state.pkl \
  --assignments-output results_incremental/ag_news_90_assignments.csv \
  --coordinates-output results_incremental/ag_news_90_coordinates.csv \
  --tree-output results_incremental/ag_news_90_tree.json \
  --plot-output results_incremental/ag_news_90_scatter.png
```

## 신규 10% 업데이트

```bash
python incremental_clustering.py update \
  --state results_incremental/ag_news_90.state.pkl \
  --input-json results_test/ag_news_embeddings_4x1000.json \
  --start 3600 --limit 400 \
  --state-output results_incremental/ag_news_100.state.pkl \
  --assignments-output results_incremental/ag_news_100_assignments.csv \
  --coordinates-output results_incremental/ag_news_100_coordinates.csv \
  --plot-output results_incremental/ag_news_100_scatter.png
```

전체 재클러스터링 사이의 증분 시각화는 저장된 PCA + UMAP 모델의 `transform`으로
신규 점만 투영하므로 기존 점의 좌표가 움직이지 않는다. 전체 재클러스터링이
발생하면 시각화 모델과 모든 좌표도 다시 계산한다. UMAP의 densMAP은 신규 점
변환을 지원하지 않으므로 증분 모드에서는 `densmap=False`가 사용된다.

## Online center와 주기적 전체 갱신

증분 update 한 번을 center update 한 번으로 계산한다. 신규 batch가 들어오면 각
계층 노드에서 `membership ** m` 가중합과 가중치를 누적하고, 구면 FCM center를
즉시 다시 정규화한다. 기본 주기는 다음과 같다.

- center update 10회: 누적된 모든 문서의 membership과 거리 임계값 재계산
- 전체 membership 재계산 3회: 누적 데이터 전체 재클러스터링
- 전체 재클러스터링 시: 시각화 PCA와 UMAP도 다시 학습하고 모든 좌표 재계산
- 신규 batch noise 비율이 30%를 초과하면 주기를 기다리지 않고 긴급 전체
  재클러스터링 및 재시각화

초기 fit에서 주기를 변경할 수 있다.

```bash
python incremental_clustering.py fit \
  ... \
  --center-updates-before-membership-refresh 10 \
  --membership-refreshes-before-recluster 3
```

state에는 center의 fuzzy sufficient statistics와 다음 실행까지 남은 카운터가
저장된다. 기존 version 1 state는 첫 update에서 기존 전체 임베딩으로 통계를
복원한다.

기본 재클러스터링 기준은 신규 배치 노이즈 비율 30% 초과다. `--noise-threshold 0.01`처럼 업데이트 명령에서 바꿀 수 있다.

## ID 규칙

문서와 클러스터·좌표를 다시 연결하려면 ID가 필요하다. 입력 레코드에서 다음 순서로 ID를 선택한다.

1. `id`
2. `resource`
3. 둘 다 없으면 원본 배열 인덱스

AG News처럼 ID가 없는 파일은 `--start`를 사용해 원본 인덱스를 보존한다. 별도 신규 파일을 사용할 때는 문서의 `id`를 넣거나, 인덱스 ID를 쓸 경우 `--id-offset`으로 기존 ID와 겹치지 않게 해야 한다.
