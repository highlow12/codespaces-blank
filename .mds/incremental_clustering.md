# 증분 클러스터링 사용법

현재 알고리즘의 전체 명세는 [`CURRENT_ALGORITHM.md`](CURRENT_ALGORITHM.md)를
참조한다. 이 문서는 명령 예시와 증분 동작을 보충한다.

`incremental_clustering.py`는 자동 선택된 클러스터링 PCA + 구면 FCM과
자동 선택된 시각화 PCA + UMAP 모델을 상태 파일로 저장하고, 이후 임베딩을
기존 클러스터에 배정한다. 신규 배치의 natural-noise 비율은 최소 표본 수 단위로
누적한 뒤 EWMA로 평활화한다. 평활 비율이 `--noise-threshold`를 초과하면 누적
전체 데이터로 자동 재클러스터링한다.

초기 `fit` 명령의 기본 동작은 클러스터링 PCA 후보(32부터)와 시각화 PCA 후보(16부터)를
각각 k-NN 보존율로 선택하는 것이다. 선택된 실제 차원은 상태 파일의
`pca_components_selected`, `visual_pca_components_selected`에 저장된다. 고정
차원이 필요하면 `--pca-components N`, `--visual-pca-components N`을 지정한다.

AG News 검증에서는 라벨별 1,000개가 묶여 있는 원본 특성을 고려해 각 라벨에서 100개씩 신규 배치로 추출했다. 따라서 초기 3,600개와 신규 400개 모두 네 라벨을 균등하게 포함한다.

## 초기 90% 학습

```bash
python incremental_clustering.py fit \
  --input-json results_test/ag_news_embeddings_4x1000.json \
  --start 0 --limit 3600 \
  --dataset-sample-size 800 \
  --fast \
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

전체 재클러스터링 사이의 증분 시각화는 저장된 선택 PCA 접두 부분 + UMAP 모델의
`transform`으로 신규 점만 투영하므로 기존 점의 좌표가 움직이지 않는다. 전체
재클러스터링이 발생해도 시각화 PCA·UMAP과 기존 좌표는 유지한다. UMAP의
densMAP은 신규 점 변환을 지원하지 않으므로 증분 모드에서는 `densmap=False`가
사용된다.

## 온라인 중심과 선택적 membership 갱신

증분 `update` 한 번을 중심 업데이트 한 번으로 계산한다. 신규 배치가 들어오면 각
계층 노드에서 `membership ** m` 가중합과 가중치를 누적하고, 구면 FCM 중심을
즉시 다시 정규화한다. 기본 주기는 다음과 같다.

- 중심 업데이트 10회: 마지막 갱신 이후 충분히 이동한 중심을 찾고, 해당 중심의
  저장 fuzzy weight가 큰 문서와 신규·교체 문서만 membership 재계산
- 선택 문서의 compact contribution만 delta로 교체하고 나머지 문서의
  contribution과 assignment는 그대로 공유
- 거리 임계값과 계층 Xie-Beni 지수는 저장된 PCA 투영값과 fuzzy weight로
  계산하므로 전 문서 membership을 다시 만들지 않음
- 선택 갱신 시 마지막 전체 재클러스터링 직후보다 계층 가중 Xie-Beni 지수가
  5% 이상 악화되면 누적 데이터 전체 재클러스터링
- 전체 재클러스터링 시: 시각화 PCA·UMAP과 좌표는 유지
- 신규 natural-noise 표본을 20개 이상 모으면 배치 비율의 EWMA를 갱신
- EWMA가 5%를 초과하면 주기를 기다리지 않고 긴급 전체 재클러스터링
- EWMA가 2.5% 이하로 내려가야 드리프트 경보를 해제하는 hysteresis 적용
- 전체 재클러스터링 후 3회 업데이트 동안 noise/XB 재클러스터링을 억제

초기 `fit` 명령에서 주기를 변경할 수 있다.

```bash
python incremental_clustering.py fit \
  ... \
  --center-updates-before-membership-refresh 10 \
  --membership-refresh-min-center-movement 0.01 \
  --membership-refresh-min-influence 0.05 \
  --max-xb-relative-degradation 0.05 \
  --drift-min-samples 20 \
  --drift-ewma-alpha 0.30 \
  --noise-threshold 0.05 \
  --noise-release-threshold 0.025 \
  --recluster-cooldown-updates 3
```

영향도는 `마지막 membership 갱신 이후 중심 이동량 × membership ** m`으로
계산한다. 기본적으로 중심 이동이 `0.01` 미만인 클러스터는 무시하고, 영향도가
`0.05` 이상인 문서만 다시 계산한다. 신규·교체 문서는 항상 선택된다. 기존
상태의 전 문서 갱신 동작이 필요하면 초기 적합 시 `--full-membership-refresh`를
지정한다.

상태 파일에는 중심의 퍼지 충분통계량, 문서별 기여도, 다음 실행까지 남은
카운터, 마지막 membership 갱신 중심 스냅샷과 EWMA·경보·cooldown 상태가
저장된다. 상태 버전은 7이다. 기존 버전
1~6 상태 파일은 기존 즉시 판정 동작(`min_samples=1`, `alpha=1`, cooldown 없음)을
유지하도록 마이그레이션하며, 첫 `update`에서 필요한 문서별 기여도를 복원한다.

각 업데이트 요약에는 `center_movement_mean/max`,
`cluster_occupancy_change`, `assignment_change_rate`가 포함된다. 작은 배치는
`drift_pending_samples`에 누적되며, 실제 판정 여부와 EWMA 값은
`drift_evaluated`, `drift_smoothed_noise_ratio`로 확인한다. 선택 갱신 범위는
`membership_refresh_scope`, `membership_refresh_sample_count`,
`membership_refresh_skipped_count`로 확인한다.

각 `update`는 `--batch-id`를 지정할 수 있다. 같은 ID와 같은 입력을 다시 실행하면
기존 상태를 변경하지 않고 replay 요약만 반환한다. 같은 ID를 다른 임베딩·메타데이터와
함께 사용하면 오류로 중단한다. ID를 생략하면 입력 내용의 fingerprint로 자동 ID를
만들어 동일 batch의 재실행을 멱등 처리한다.

상태 파일은 SHA-256 checksum envelope와 atomic replace로 저장된다. CLI update는
상태 파일 옆의 lock을 잡고 load→update→save를 수행하므로 동시 update가 서로의
결과를 덮어쓰지 않는다.

```bash
./.venv/bin/python incremental_clustering.py update \
  --state results_incremental/model.state.pkl \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --batch-id gemini-batch-2026-08-07 \
  --state-output results_incremental/model.updated.state.pkl
```

## 변경된 노트 교체

`update` 입력에 상태 파일에 이미 존재하는 `id`가 포함되면 새 문서로 중복 추가하지
않고 해당 ID의 기존 임베딩·메타데이터·할당 결과·고정 좌표를 교체한다. 기존
임베딩의 `membership ** m` 기여도는 중심 통계에서 제거한 뒤 새 임베딩의
기여도로 대체하므로 중심이 같은 노트를 두 번 반영하지 않는다. 입력에 기존
ID와 새 ID가 섞여 있어도 같은 규칙이 각각 적용된다.

실제로 새 문서를 추가하는 경우에만 `--id-offset`으로 생성 인덱스 ID가 기존
ID와 겹치지 않도록 한다.

기본 긴급 재클러스터링 진입 기준은 평활 natural-noise 비율 5% 초과다.
`--noise-threshold 0.01 --noise-release-threshold 0.005`처럼 업데이트 명령에서
진입·해제 기준을 함께 바꿀 수 있다.

## FCM 클러스터 개수 자동 선택

각 계층 노드의 FCM은 기본적으로 `K=2`부터 시작해 다음 지표를 계산한다.

- XB(Xie-Beni): 낮을수록 좋음
- 실루엣: 높을수록 좋음
- 재시작 안정성(유효 재시작 간 평균 ARI): 높을수록 좋음
- 수정 분할 계수: 높을수록 좋음
- 분할 계수와 분할 엔트로피: 진단용으로 저장

기본 `multi_metric` 방식에서는 별도 노이즈를 만들지 않고 모든 샘플을 최대
소속도의 클러스터에 배정한 뒤 지표를 계산한다. 단, 최소 자식 크기를
충족하지 못한 후보는 선택 대상에서 제외한다.

500건 이상 노드에서는 기본적으로 20% 독립 표본을 최대 5개 평가하고 같은
K가 3표를 얻으면 그 K만 전체 노드에서 적합한다. 합의 실패 또는 전체 적합
실패 시 아래의 전체 K 탐색으로 자동 복귀한다. 작은 노드는 처음부터 전체
탐색을 사용하며 `--exact-k-selection`으로 표본 합의를 끌 수 있다.

XB가 직전 유효 K보다 처음으로 악화되면 그 지점에서 K를 기본 두 번 더
평가한 뒤 탐색을 중단한다. 평가한 유효 후보 전체에 대해 네 선택 지표의
순위를 `0~1` 선호도 점수로 바꾼다. 동률은 평균 순위를 사용하며 XB 40%,
실루엣 25%, 재시작 안정성 25%, 수정 PC 10%로 합산한 `selection_score`가
가장 높은 `K`를 선택한다. 분할 계수·분할 엔트로피 원값과 정규화 엔트로피는
결과에 함께 저장하지만 기본 점수에는 넣지 않는다.

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
후보의 모든 지표와 `selection_score`는 트리 JSON의 `candidate_metrics`에
저장된다.

`--selection-method xie_beni`는 XB 상대 개선량이
`--min-xb-relative-improvement`보다 작아지면 중단하는 호환 모드다.
`silhouette`와 `knee` 선택 방식도 지원한다.

## ID 규칙

문서와 클러스터·좌표를 다시 연결하려면 ID가 필요하다. 입력 레코드에서 다음 순서로 ID를 선택한다.

1. `id`
2. `resource`
3. 둘 다 없으면 원본 배열 인덱스

AG News처럼 ID가 없는 파일은 `--start`를 사용해 원본 인덱스를 보존한다. 별도 신규 파일을 사용할 때는 문서의 `id`를 넣거나, 인덱스 ID를 쓸 경우 `--id-offset`으로 기존 ID와 겹치지 않게 해야 한다.
