# 현재 알고리즘 명세

이 문서는 현재 `main` 작업 트리의 기본 실행 경로를 설명한다. 기준 명령은
`incremental_clustering.py fit`과 `incremental_clustering.py update`이며,
실제 구현과 옵션 기본값이 바뀌면 이 문서를 함께 갱신한다.

기존 `PROJECT_DECISION_HISTORY.md`, `PCA_DIMENSION_SELECTION.md`와 각 실험
문서는 과거 평가와 설계 결정을 보존한다. 과거 문서에 나오는 고정 PCA-256,
고정 PCA-64 또는 이전 지표 가중치는 현재 기본값이 아닐 수 있다.

## 1. 구현 기준 파일

- `incremental_clustering.py`: 초기 적합, 증분 업데이트, 상태 저장, CLI
- `hierarchical_fcm.py`: 재귀 계층 분할과 할당 결과 생성
- `fcm_core.py`: 정규화 PCA와 구면 FCM
- `fcm_validity.py`: 후보 K 평가와 선택
- `fcm_document_classification.py`: core·boundary·noise 판정
- `pca_dimension_search.py`: PCA 접두 부분과 k-NN 보존율 평가
- `visualization_pca_dimension_selection.py`: UMAP 앞 시각화 PCA 선택
- `umap_projection.py`: PCA + UMAP 학습과 신규 점 변환
- `fast_fcm.py`: `--fast`용 제한된 탐색 경로

`full_pipeline.py`는 별도의 실험·검증용 래퍼이며, CLI 기본값이
`incremental_clustering.py`와 다를 수 있다.

## 2. 전체 흐름

```text
JSON 임베딩과 메타데이터
  → 입력 검증 및 ID 정렬
  → 정규화 PCA 차원 자동 선택
  → 선택된 PCA 공간에서 재귀 구면 FCM
  → 각 노드의 K 선택과 노이즈·경계 판정
  → 계층별 소프트 소속도와 경로 소속도 생성
  → 시각화용 PCA 차원 자동 선택
  → 약지도 UMAP-2 학습
  → 상태 파일에 모델·중심·좌표 저장
```

증분 업데이트에서는 저장된 PCA·계층 중심·UMAP을 재사용한다.

```text
신규 배치
  → 저장된 클러스터링 PCA로 변환
  → 고정된 계층 중심에 소프트 배정
  → 저장된 UMAP으로 신규 좌표만 transform
  → 누적 퍼지 충분통계량으로 중심 온라인 갱신
  → 주기적으로 중심 영향도가 큰 문서만 소속도·contribution 갱신
  → 저장 contribution으로 거리 임계값·계층 XB 갱신
  → 노이즈 또는 XB 조건 충족 시 누적 데이터 전체 재클러스터링
```

## 3. 실행 방법

### 3.1 초기 적합

```bash
python incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --state-output results/model.state.pkl \
  --assignments-output results/model_assignments.csv \
  --coordinates-output results/model_coordinates.csv \
  --tree-output results/model_tree.json \
  --plot-output results/model_scatter.png
```

`fit`은 클러스터링 모델과 시각화 모델을 모두 학습한다. 출력 경로를 생략한
할당·좌표·트리 파일은 상태 경로의 stem을 기준으로 자동 생성된다.

반복적인 클러스터링 실험에서는 다음처럼 UMAP을 생략할 수 있다.

```bash
python incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --state-output results/scout.state.pkl \
  --fast \
  --skip-visualization
```

이 상태는 클러스터링 검사 전용이다. `update`에 사용하려면
`--skip-visualization` 없이 최종 상태를 다시 적합해야 한다.

### 3.2 증분 업데이트

```bash
python incremental_clustering.py update \
  --state results/model.state.pkl \
  --input-json new_embeddings.json \
  --state-output results/model_updated.state.pkl \
  --assignments-output results/model_updated_assignments.csv \
  --coordinates-output results/model_updated_coordinates.csv
```

신규 문서의 좌표는 기존 UMAP 모델로 `transform`하므로 기존 문서의 좌표는
움직이지 않는다. 전체 재클러스터링이 발생해도 시각화 PCA·UMAP 모델과 기존
좌표는 유지한다.

### 3.3 입력 ID와 표본 추출

입력은 최상위 배열 또는 `{"records": [...]}` 형식의 JSON을 지원한다. 각
레코드는 `embedding`을 반드시 가져야 한다. ID는 다음 순서로 정한다.

1. 레코드의 `id`
2. `id`가 없을 때 `resource`
3. 둘 다 없을 때 `--id-offset + 원본 배열 인덱스`

선택된 배치 안에서 ID는 유일해야 한다. `--start`와 `--limit`은 원본 배열
범위를 선택하고, `--dataset-sample-size`는 적합 전에 복원 없이 무작위 표본을
추출한다. 표본 시드가 생략되면 `--seed`를 사용하며, 임베딩과 메타데이터의
행 정렬은 유지된다.

## 4. 기본 CLI 설정

다음은 `incremental_clustering.py fit`의 기본값이다. `--pca-components`와
`--visual-pca-components`를 지정하면 각각 해당 자동 선택을 끈다.

| 영역 | 기본값 |
|---|---|
| 최대 계층 깊이 | `4` |
| 최소 노드 크기 | `60` |
| 최소 자식 크기 | `20` |
| 후보 K | `2`부터 `4`까지, 노드 크기로 추가 제한 |
| 최소 최대 소속도 | `0.20` |
| 최대 소속도 차이 | `0.10` |
| 전역 강제 노이즈 비율 | `0.01` |
| 거리 이상치 배수 | `3.5` |
| K 선택 방식 | `multi_metric` |
| 퍼지화 지수 `m` | `2.0` |
| FCM 최대 반복 | `200` |
| FCM 수렴 허용오차 | `1e-6` |
| 시드 | `42` |
| 신규 배치 긴급 재클러스터 기준 | 자연 노이즈 비율 `0.05` 초과 |
| 중심 영향 기반 선택 소속도 갱신 | 중심 업데이트 `10`회 |
| 선택 대상 최소 중심 이동 | `0.01` |
| 선택 대상 최소 영향도 | `0.05` |
| 계층 XB 악화 재클러스터 기준 | 기준 XB 대비 `0.05` 이상 |
| UMAP 이웃 수 | `15` |
| UMAP `min_dist` | `0.02` |
| UMAP 거리 | `cosine` |
| UMAP `spread` | `0.85` |
| 약지도 목표 가중치 | `0.01` |
| 증분 UMAP `densmap` | `False` |

`--visual-densmap`으로 요청할 수 있지만 densMAP은 신규 점 변환을 지원하지
않으므로 증분 상태에서는 경고 후 비활성화된다.

## 5. PCA 차원 선택

### 5.1 공통 전처리

원본 임베딩 행마다 L2 정규화를 먼저 적용한다. PCA는 정규화된 원본에 한 번
적합하고, PCA 결과도 다시 L2 정규화한다.

```text
Xn = L2Normalize(X)
Z = PCA.fitTransform(Xn)
Zprefix = L2Normalize(Z[:, :d])
```

PCA는 최대 후보 폭으로 한 번만 적합하며, 후보마다 같은 PCA 결과의 접두
부분을 잘라 사용한다. 이후 배치에도 같은 PCA와 선택된 접두 부분을 재사용한다.

### 5.2 클러스터링 PCA

기본 자동 선택은 다음 설정을 사용한다.

- 후보 시작 차원: `32`
- 후보 증가 폭: `32`
- 최대 후보: `512`와 표본 수·임베딩 차원 중 작은 값
- 이웃 수: `k=15`, `k=30`
- 후보 점수: 원본 정규화 임베딩과 정규화 PCA 공간의 평균 k-NN 교집합 비율
- 정지 기준: 평균 보존율 증가량이 처음으로 `0.05` 미만이면 직전 차원 선택
- 모든 증가량이 기준 이상이면 마지막 후보 선택

실제 입력이 작으면 후보 폭, 표본 수, 임베딩 차원에 맞춰 안전하게 줄인다.
선택된 차원과 후보별 지표는 상태 설정의
`pca_components_selected`, `pca_selection`에 저장된다.

### 5.3 시각화 PCA

시각화는 클러스터링 PCA와 분리된 자동 선택을 사용한다.

- 후보 시작 차원: `16`
- 후보 증가 폭: `16`
- 최대 후보: `512`와 표본 수·임베딩 차원 중 작은 값
- 각 후보: 정규화 PCA 접두 부분 → 동일 설정의 UMAP-2
- 점수: 원본 정규화 임베딩과 UMAP-2 좌표의 k-NN 교집합 비율
- 정지 기준: 보존율 증가량이 처음 `0.05` 미만이면 직전 PCA 차원 선택

모든 후보가 기준을 만족하면 최대 후보를 선택한다. 선택된 PCA 차원과 UMAP
모델은 상태에 저장되며 `visual_pca_components_selected`에서 확인할 수 있다.

## 6. 재귀 구면 FCM

### 6.1 FCM 계산

클러스터링 PCA 결과를 다시 단위 벡터로 취급한다. 각 후보 K에 대해 다음을
수행한다.

1. `kmeans++`로 중심을 초기화하고 중심을 L2 정규화한다.
2. 중심까지의 유클리드 거리로 소프트 소속도를 초기화한다.
3. `membership ** m`을 가중치로 중심을 계산한다.
4. 각 중심을 다시 L2 정규화한다.
5. 새 거리로 소속도를 갱신한다.
6. 최대 소속도 변화가 `tol`보다 작으면 종료한다.

입력과 중심이 모두 단위 구면에 있으므로 유클리드 거리 순위는 코사인
거리 순위와 대응한다. 별도의 코사인 FCM 분기를 사용하지 않는다.

기본 일반 경로는 재시작을 `10`회 시도하고, 최대 `30`회까지 시도할 수
있다. 유효한 재시작은 모든 클러스터가 최소 자식 크기를 만족하고 중심 간
최소 거리가 `1e-3` 이상인 결과다. 유효한 결과 중 FCM 목적 함수가 가장
작은 결과를 선택하며, 유효 재시작 사이의 평균 ARI를 안정성 지표로 저장한다.

### 6.2 후보 K와 기본 `multi_metric` 선택

각 노드의 후보 범위는 다음과 같다.

```text
maximum_k = min(max_clusters, floor(node_size / min_child_size))
k = min_clusters ... maximum_k
```

기본 `multi_metric` 후보 평가는 모든 샘플을 최대 소속도 클러스터에
배정한 상태로 계산한다. 후보별로 XB, 실루엣, 재시작 안정성, 분할 계수,
분할 엔트로피, 목적 함수, 중심 거리와 클러스터 크기를 저장한다. 후보가
유효하려면 최소 두 개의 자식, 최소 자식 크기, 유효 재시작, 유한한 XB·실루엣,
중심 분리 조건을 만족해야 한다.

XB가 직전 유효 후보보다 처음 악화되면 기본적으로 K를 두 개 더 확인한 뒤
탐색을 멈춘다. 유효 후보의 지표를 평균 순위 기반 `0~1` 선호도 점수로
바꾸고 다음 가중치로 합산한다.

| 지표 | 방향 | 가중치 |
|---|---|---:|
| XB | 낮을수록 좋음 | `0.40` |
| 실루엣 | 높을수록 좋음 | `0.25` |
| 재시작 안정성(평균 ARI) | 높을수록 좋음 | `0.25` |
| 수정 분할 계수 | 높을수록 좋음 | `0.10` |

동률 순위는 평균 순위를 사용한다. 분할 엔트로피와 정규화 분할 엔트로피는
결과에 저장하지만 기본 점수에는 포함하지 않는다. 최종 K는 `selection_score`
가 가장 높은 후보이며, 남은 동률은 XB·안정성·실루엣·수정 분할 계수·작은 K
순으로 결정한다.

`silhouette`, `knee`, `xie_beni` 선택 방식도 CLI에서 사용할 수 있다. 특히
`min-split-silhouette`는 기본 `multi_metric` 선택의 분할 중단 조건으로는
사용되지 않고, 실루엣 기반 선택 방식에서 유효하다.

`multi_metric`을 사용하는 500건 이상 노드는 기본적으로 전체 후보 탐색 전에
표본 합의 선택을 수행한다. 서로 다른 seed의 20% 표본을 최대 5개 만들고,
저비용 `n_init=3` 선택에서 같은 K가 3표를 얻으면 그 K 하나만 전체 노드에서
기본 `n_init=10`으로 적합한다. 3표 합의가 없거나 선택 K의 전체 적합이
유효하지 않으면 기존의 전체 K 탐색으로 자동 복귀한다. 500건 미만 노드와
`silhouette`, `knee`, `xie_beni` 방식은 기존 exact 선택을 유지한다.
CLI의 `--exact-k-selection`은 표본 합의를 비활성화한다.

### 6.3 재귀와 조기 중단

루트에서 시작해 선택된 생존 클러스터만 다음 계층으로 전달한다. 다음 경우에
재귀를 중단한다.

- 최대 깊이 `4`에 도달
- 현재 노드가 최소 노드 크기 `60`보다 작음
- 유효한 K 후보가 없음
- 노이즈 필터 후 생존 자식이 두 개 미만
- 비기본 선택 방식에서 분할 실루엣이 임계값보다 낮음

내부 계층 ID는 각 부모 아래에서 `0`부터 시작한다. 전역 경로는
`0/1/2`처럼 `/`로 연결한다. 시각화 표시 라벨은 이를 `1-2-3`처럼
1부터 시작하는 형태로 바꾼다.

## 7. 노이즈·경계·소프트 결과

### 7.1 자연 노이즈와 경계

각 샘플의 최대 소속도가 `0.20` 미만이고, 최대·두 번째 소속도 차이가
`0.10` 미만이면 경계 후보가 된다. 해당 샘플이 배정된 중심에서 다음
강건 거리 임계값보다 멀면 자연 노이즈로 판정한다.

```text
threshold = median(cluster_distances)
          + 3.5 * 1.4826 * MAD(cluster_distances)
```

클러스터 표본이 4개 미만이거나 MAD가 0에 가까우면 거리 임계값은 무한대로
두므로, 소속도 조건만으로 노이즈가 되지 않는다. 경계 후보지만 임계값 안에
있는 샘플은 `boundary`, 그 밖의 샘플은 `core`다. 자연 노이즈는 하위 재귀에
전달하지 않는다.

### 7.2 전역 강제 노이즈

각 샘플에 대해 낮은 신뢰도, 작은 상위 두 소속도 차이, 중심 거리의 백분위
순위를 결합한 `noise_score`를 계산한다. 기본적으로 전체 문서의 상위 1%를
강제 노이즈로 추가한다. 동점은 문서 ID 순으로 결정한다.

결과에서는 자연 노이즈와 강제 노이즈를 각각
`is_natural_noise`, `is_forced_noise`로 구분한다. 강제 노이즈만 해당하는
샘플의 `noise_level`은 `0`으로 표시된다.

### 7.3 소프트 소속도와 경로 소속도

각 계층의 `level_N_membership_K`는 해당 노드의 지역 FCM 소속도다. 서로
다른 부모 아래의 지역 소속도를 직접 비교하지 않는다. 부모에 도달할 확률을
곱한 `level_N_path_membership_*`를 사용한다.

```text
P(0/1) = P(0) × P(0/1 | 0)
```

이 경로 소속도는 같은 깊이의 전역 경로를 비교하는 데 사용되며, 약지도 UMAP의
우선 목표이기도 하다.

## 8. 시각화

시각화용 UMAP 목표는 다음 우선순위로 만든다.

1. 계층 경로 소프트 소속도
2. 평면 소프트 소속도
3. 소프트 값이 없을 때의 계층 표시 라벨

소프트 목표는 행별 L2 정규화 후 `euclidean` 목표 거리로 UMAP에 전달하며,
기본 목표 가중치는 `0.01`이다. 이 가중치는 원본 임베딩의 기하 구조를
대체하지 않고 경계 정보를 약하게 보조한다.

기본 UMAP 설정은 `n_components=2`, `n_neighbors=15`, `min_dist=0.02`,
`metric=cosine`, `spread=0.85`, 시드 `42`다. 모델은 초기 적합 때 한 번만
학습하고, 신규 문서는 같은 PCA 접두 부분과 UMAP 모델의 `transform`으로
투영한다.

색상은 `--color-by cluster`만 지원한다. 태그·정답 범주 색상은 사용하지
않는다. 최상위 표시 라벨마다 다른 색조를 쓰고, 하위 경로는 같은 색조의
명도 차이로 표시한다. `noise`와 `*-noise`는 회색 `#9aa0a6`다.

## 9. 증분 업데이트

### 9.1 중심 온라인 갱신

초기 적합 후 상태에는 각 문서가 각 계층 노드에 기여한 퍼지 충분통계량을
저장한다.

```text
weight = membership ** m
weighted_sum = weight * projected_embedding
center = L2Normalize(sum(weighted_sum) / sum(weight))
```

매 `update`마다 신규 문서의 기여도를 누적하고 해당 노드 중심을 즉시 다시
정규화한다. 기존 ID가 들어오면 기존 문서의 기여도를 제거하고 새 기여도로
교체하므로 같은 문서를 두 번 세지 않는다.

### 9.2 선택 소속도 갱신과 재클러스터링

- 중심 업데이트가 10회 누적되면 마지막 갱신 중심과 현재 중심의 누적 이동을
  계산한다.
- 이동량이 `0.01` 이상인 중심에 대해
  `중심 이동량 × 저장 membership ** m >= 0.05`인 문서만 다시 배정한다.
- 신규·교체 문서는 중심 이동과 관계없이 항상 다시 배정한다.
- 선택 문서의 assignment와 compact contribution만 delta로 교체한다.
- 거리 임계값과 계층 가중 XB는 저장된 PCA 투영값과 fuzzy weight로 계산해
  나머지 문서의 membership을 다시 계산하지 않는다.
- 구형 상태는 호환성을 위해 기존 전 문서 갱신을 유지한다.
- 현재 XB가 초기 기준 XB보다 `5%` 이상 악화되면 누적 데이터 전체를
  재클러스터링한다.
- 자연 노이즈는 최소 20개 표본 단위의 EWMA로 평활화하며, 기본 `5%` 진입과
  `2.5%` 해제 threshold를 사용한다. 경보가 활성화되면 전체 재클러스터링한다.
- 재클러스터링 뒤 3회 업데이트 동안 noise/XB 트리거를 억제한다.
- 강제 노이즈 1%는 긴급 재클러스터링 판단에는 포함하지 않고, 결과 표시에
  적용한다.
- 재클러스터링해도 시각화 PCA·UMAP 모델과 기존 좌표는 다시 학습하지 않는다.

업데이트 결과에는 `reclustered`, `membership_refreshed`,
`membership_refresh_sample_count`, `membership_refresh_skipped_count`,
`emergency_recluster`, `xb_relative_degradation`, `new_noise_ratio` 등이 기록된다.

## 10. 빠른 탐색 모드

`--fast`는 최종 실험보다 반복 개발에 적합한 제한 경로다.

1. 노드마다 최대 `600`개를 시드 기반으로 표본 추출한다.
2. `m=[2.0, 1.8, 1.6, 1.4]`를 순서대로 조사해 안정적인 퍼지화 지수를 찾는다.
3. 표본에서 최대 K `4`까지 탐색하고, 점수가 좋은 K 하나만 전체 노드에서
   정밀화한다.
4. 정밀화 재시작 안정성이 `0.85`보다 낮으면 재시작 수를 늘리며 최대 `10`회
   수준까지 확장한다.
5. 반환되는 최종 후보의 라벨·중심·지표는 전체 노드 기준으로 계산한다.

`--fast`는 일반 전수 후보 탐색보다 빠르지만, 최종 결과를 확정할 때는
기본 전수 경로와 후보 지표를 비교하는 것이 좋다.

## 11. 출력 형식

### 11.1 할당 CSV

기본 할당 결과에는 다음 필드가 포함된다.

- 입력 메타데이터와 `id`
- `level_1_cluster`부터 `level_4_cluster`까지의 0-based 계층 ID
- `level_N_membership_K` 지역 소프트 소속도
- `level_N_path_membership_*` 전역 경로 소속도
- `cluster`: 마지막 유효 계층의 ID, 노이즈는 `-1`
- `cluster_path`: `0/1/2`, 노이즈 경로는 `0/1/noise`
- `is_noise`, `is_natural_noise`, `is_forced_noise`, `is_boundary`
- `document_type`: `core`, `boundary`, `noise`
- `noise_score`, `boundary_level`, `noise_level`, `leaf_level`

할당과 좌표는 행 순서가 아니라 `id`를 기준으로 결합한다.

### 11.2 트리 JSON

트리 JSON은 `config`, `summary`, `root`를 최상위에 둔다. 노드마다 선택된 K,
실루엣, XB, 분할 지표, `selection_score`, 후보별 지표, 노이즈·경계 개수,
중단 사유와 자식 경로를 저장한다. `summary`에는 표본 수, 실제 PCA 차원,
요청·도달 계층, 노드·리프 수, 자연·강제 노이즈 수, 경계·core 수, 계층별
노이즈 수, 실행 시간이 포함된다.

### 11.3 상태 파일

pickle 상태에는 다음을 포함한다.

- 누적 임베딩·메타데이터·할당·좌표
- 재사용 가능한 클러스터링 PCA와 계층 중심
- 계층 트리와 전체 설정
- 시각화 PCA와 UMAP 모델
- 노드별 누적 중심 통계량
- ID별 중심 기여도
- 중심 업데이트·소속도 재계산·전체 재클러스터링 카운터

## 12. 현재 기본값과 과거 문서의 구분

현재 기본값을 한 줄로 요약하면 다음과 같다.

> 자동 선택 클러스터링 PCA(기본 후보 32부터) + 재귀 구면 FCM
> (`multi_metric`, K 최대 4) + 자동 선택 시각화 PCA(기본 후보 16부터)
> + 소프트 경로 목표 가중치 0.01 + 고정 좌표 증분 업데이트

PCA-256과 PCA-64는 과거 데이터에서 평가한 고정 후보이며, 현재 코드의
자동 선택 결과를 대체하지 않는다. 지표 가중치도 과거 문서의 XB·PC·PE
조합이 아니라 이 문서 6.2절의 XB·실루엣·재시작 안정성·수정 PC 조합을
사용한다.
