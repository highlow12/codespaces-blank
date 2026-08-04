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

증분 시각화는 초기 PCA + UMAP 모델을 저장하고 `transform`으로 신규 점만 투영한다. 따라서 기존 점의 좌표는 움직이지 않는다. UMAP의 densMAP은 신규 점 변환을 지원하지 않으므로 증분 모드에서는 `densmap=False`가 사용된다.

기본 재클러스터링 기준은 신규 배치 노이즈 비율 30% 초과다. `--noise-threshold 0.01`처럼 업데이트 명령에서 바꿀 수 있다.

## FCM 클러스터 개수 자동 선택

각 계층 노드의 FCM은 기본적으로 `K=2`부터 시작해 다음 네 지표를 계산한다.

- XB(Xie-Beni): 낮을수록 좋음
- Partition Coefficient: 높을수록 좋음
- Partition Entropy: 낮을수록 좋음

기본 `multi_metric` 방식에서는 별도 noise를 만들지 않고 모든 샘플을 최대
membership의 클러스터에 배정한 뒤 지표를 계산한다. 단, 최소 child 크기를
충족하지 못한 후보는 선택 대상에서 제외한다.

XB가 직전 K보다 처음으로 악화되면 그 지점에서 K를 기본 두 번 더 늘려
평가한 뒤 탐색을 중단한다. 평가한 후보 전체에 대해 XB, modified PC,
normalized PE의 순위를 `0~1` 선호도 점수로 바꾼다. 동률은 평균 순위를
사용하며 XB 50%, modified PC 25%, normalized PE 25%로 합산한
`selection_score`가 가장 높은 `K`를 선택한다. 따라서 하나의 극단값이
다른 후보 사이의 차이를 압축하지 않는다. PC와 PE 원값도 결과에 함께
저장한다. Silhouette은 진단 출력에는 남지만 `multi_metric` 방식의 선택,
탐색 중단, 계층 분할 중단에는 사용하지 않는다.

```text
relative_improvement(K) = (XB(K-1) - XB(K)) / abs(XB(K-1))
```

XB 악화 후 추가로 확인할 K 개수는 기본 2개다. 다음처럼 변경할 수 있다.

```bash
python incremental_clustering.py fit \
  ... \
  --selection-method multi_metric \
  --xb-worsening-patience 3
```

XB가 악화되지 않으면 `--max-clusters`까지 평가한다. 각
후보의 모든 지표와 `selection_score`는 tree JSON의 `candidate_metrics`에
저장된다.

## ID 규칙

문서와 클러스터·좌표를 다시 연결하려면 ID가 필요하다. 입력 레코드에서 다음 순서로 ID를 선택한다.

1. `id`
2. `resource`
3. 둘 다 없으면 원본 배열 인덱스

AG News처럼 ID가 없는 파일은 `--start`를 사용해 원본 인덱스를 보존한다. 별도 신규 파일을 사용할 때는 문서의 `id`를 넣거나, 인덱스 ID를 쓸 경우 `--id-offset`으로 기존 ID와 겹치지 않게 해야 한다.
