# HDBSCAN leaf 기반 bottom-up 계층 클러스터링 연구

> 상태: 유망한 연구 트랙. 현재 운영 기본값인 계층형 구면 FCM을 아직 대체하지
> 않는다. 이 문서는 2026-08-15의 탐색 실험, 구현, 한계와 다음 검증 계획을
> 보존한다.

## 1. 연구 질문

이 연구는 다음 가능성을 검증한다.

> 저차원 의미 공간에서 HDBSCAN으로 빠르고 순수한 leaf를 발견하고, 원래 의미
> 정보가 더 많이 남아 있는 PCA 공간에서 유사한 leaf를 bottom-up으로 병합하면
> 속도와 계층 품질을 동시에 얻을 수 있는가?

기존 재귀 FCM은 소프트 소속도, 노드별 K 선택과 증분 갱신이라는 장점이 있지만,
모든 계층에서 후보 K와 재시작을 반복하므로 초기 적합 비용이 크다. 반면 HDBSCAN은
한 번의 밀도 탐색으로 leaf 후보와 noise를 빠르게 얻을 수 있다. 두 접근의 장점을
결합하기 위해 HDBSCAN은 leaf 발견에만 사용하고, 상위 의미 계층은 별도의
bottom-up 병합으로 만든다.

## 2. 데이터와 정답 계층

검증 데이터는 `dbpedia_gemini_embeddings.json.gz`다.

| 항목 | 값 |
|---|---:|
| 문서 수 | 3,000 |
| 원본 임베딩 차원 | 3,072 |
| 상위 정답 `class_hierarchy[0]` | 6개 |
| 중간 정답 `class_hierarchy[1]` | 17개 |
| leaf 정답 `class` | 18개 |
| 클래스별 문서 수 | 166~167개 |

정답 구조는 모든 문서에서 3단계이며 다음처럼 균형 있게 구성돼 있다.

```text
6개 top
  → 17개 middle
      → 18개 leaf
```

`Sports` 중간 범주만 `Athlete`와 `SportsTeam` 두 leaf를 가지며, 나머지 중간
범주는 leaf 하나와 대응한다.

## 3. 탐색 과정

### 3.1 PCA-96에서 HDBSCAN 직접 실행

첫 기준선은 다음과 같았다.

```text
원본 임베딩
  → L2 정규화
  → PCA-96
  → L2 정규화
  → HDBSCAN(min_cluster_size=20, min_samples=5)
```

| 군집 | noise | class NMI | class ARI | noise 제외 silhouette |
|---:|---:|---:|---:|---:|
| 23 | 47.50% | 0.5799 | 0.1298 | 0.2081 |

PCA 공간에 HDBSCAN을 직접 적용하면 실행은 빠르지만 절반에 가까운 문서를
noise로 처리했다. `cluster_selection_epsilon`을 `0.02~0.6`으로 높여도 결과가
바뀌지 않았고, `0.8`에서 23개가 22개로만 줄었다. `1.0`에서는 2개 군집으로
급격히 붕괴해 class NMI `0.2154`, ARI `0.0389`가 됐다. 따라서 병합 epsilon만으로
noise와 과분할을 동시에 해결하기 어렵다고 판단했다.

### 3.2 PCA 뒤 UMAP 12~20차원

밀도 계산 공간을 만들기 위해 UMAP을 2차원 시각화가 아닌 12~20차원 특징
변환으로 사용했다. HDBSCAN 설정은 다음으로 고정했다.

```text
min_cluster_size = 40
min_samples = 2
cluster_selection_method = eom
```

cosine UMAP에서는 UMAP-20이 가장 좋은 class 정합성을 보였다. 다만 cosine
거리는 입력 벡터 크기에 불변이므로 PCA 후 L2 유무가 크기 정보의 효과를 직접
검증하지 못한다. 이를 분리하기 위해 UMAP 거리를 Euclidean으로 바꿔 다시
비교했다.

| UMAP 거리 | PCA 후 L2 | 군집 | noise | class NMI | class ARI |
|---|---|---:|---:|---:|---:|
| cosine | 미적용 | 25 | 2.37% | 0.8275 | 0.7193 |
| cosine | 적용 | 25 | 4.30% | 0.8339 | 0.7343 |
| Euclidean | 미적용 | 24 | 5.77% | 0.8106 | 0.6932 |
| **Euclidean** | **적용** | **24** | **3.33%** | **0.8351** | **0.7477** |

현재 데이터에서는 PCA 결과의 크기를 유지하는 것보다 다시 단위 구면으로
정규화하는 편이 좋았다. PCA 크기 정보가 유용한 의미 신호라기보다 UMAP의
Euclidean 이웃과 밀도 계산을 흔드는 편차로 작용한 것으로 해석한다.

### 3.3 UMAP seed 반복

최종 leaf 파이프라인에서 PCA seed는 42로 고정하고 UMAP seed를 세 번 비교했다.

| UMAP seed | 군집 | noise | class NMI | class ARI |
|---:|---:|---:|---:|---:|
| 42 | 24 | 3.33% | 0.8351 | 0.7477 |
| 43 | 24 | 2.77% | 0.8164 | 0.7018 |
| 44 | 24 | 4.73% | 0.8127 | 0.6906 |
| 평균 | 24 | 3.61% | 0.8214 | 0.7134 |

세 실행 모두 24개 군집을 만들었고 noise는 5% 미만이었다. 외부 정답 지표는
양호했지만 seed 간 문서 분할 ARI는 아직 저장·측정하지 않았다. 동일한 군집 수가
동일한 분할을 뜻하지 않으므로 운영 채택 전에 partition 안정성 검증이 필요하다.

## 4. 현재 bottom-up 알고리즘

### 4.1 전체 흐름

```text
원본 임베딩 3,072차원
  → 행별 L2 정규화
  → PCA-96
  → PCA 결과 행별 L2 정규화
  → UMAP-20
      n_neighbors=30
      min_dist=0.0
      metric=euclidean
  → HDBSCAN
      min_cluster_size=40
      min_samples=2
      cluster_selection_method=eom
      prediction_data=True
  → 24개 flat cluster를 애플리케이션 leaf로 사용
  → PCA-96 공간에서 leaf별 소프트 가중 중심 계산
  → 중심 cosine 거리 기반 bottom-up 병합
  → 원하는 군집 수에서 트리 절단
```

HDBSCAN의 `eom` 결과는 HDBSCAN condensed tree의 말단 노드라는 뜻은 아니다.
이 연구에서는 선택된 24개 flat cluster를 애플리케이션 계층의 leaf로 정의한다.

### 4.2 소프트 leaf 중심

`all_points_membership_vectors()`의 문서별 leaf membership을 사용한다. 문서
`i`의 정규화 PCA 벡터를 `z_i`, leaf `c`의 소속도를 `u_ic`라고 하면 중심과
질량은 다음과 같다.

```text
mass_c = Σ_i u_ic

center_c = L2Normalize(
    Σ_i u_ic × z_i / mass_c
)
```

하드 라벨만으로 중심을 만들지 않으므로 경계 문서의 정보를 연속적인 가중치로
반영한다. noise 문서도 낮은 membership만큼만 중심에 기여한다.

### 4.3 bottom-up 병합

초기 leaf 간 거리는 정규화 중심의 cosine 거리다.

```text
d(A, B) = 1 - cosine(center_A, center_B)
```

가장 가까운 두 활성 노드를 반복해서 병합한다. 새 노드와 다른 노드 사이의
거리는 membership 질량으로 가중한 average linkage로 갱신한다.

```text
d(A∪B, C)
  = (mass_A × d(A,C) + mass_B × d(B,C))
    / (mass_A + mass_B)
```

동일 거리에서는 노드 ID 순으로 결정해 결과를 재현 가능하게 유지한다. 전체
이진 병합 트리를 만든 뒤 첫 `leaf_count - K`개 병합을 적용하면 정확히 K개의
상위 군집을 얻는다.

### 4.4 소속도와 noise 정책

부모 소속도는 자식 leaf 소속도의 합으로 만들 수 있다.

```text
P(parent | document) = Σ P(child leaf | document)
```

따라서 서로 독립적으로 학습한 지역 모델의 소속도를 비교하는 문제가 없다.
HDBSCAN hard noise `-1`은 현재 모든 상위 cut에서도 `-1`로 유지한다. 이후에는
다음 두 값을 함께 저장하는 방식을 검토한다.

- 절대 부모 소속도: noise 질량을 포함한 원래 membership 합
- 조건부 부모 소속도: non-noise 질량으로 다시 나눈 부모 간 상대 확률

## 5. bottom-up 계층 결과

seed 42, 3,000건 전체 실행 결과다. 모든 NMI/ARI는 noise `-1`을 별도 예측
라벨로 포함해 계산했다.

| 평가 레벨 | 정답 군집 수 | 예측 cut | NMI | ARI | noise |
|---|---:|---:|---:|---:|---:|
| top | 6 | 6 | 0.7162 | 0.6493 | 3.33% |
| middle | 17 | 17 | 0.7945 | 0.6421 | 3.33% |
| leaf | 18 | 18 | 0.8222 | 0.7122 | 3.33% |
| 병합 전 leaf | 18개 leaf 정답 | 24 | 0.8351 | 0.7477 | 3.33% |

24개 leaf를 18개로 병합했을 때 NMI는 `0.0129`, ARI는 `0.0355`만 감소했다.
추가 leaf 중 상당수가 정답 leaf를 의미적으로 순수하게 세분한 결과이며,
bottom-up 트리가 그 구조를 비교적 잘 복원했다는 신호다.

top 6개 cut에서는 다음 혼합이 주요 오차였다.

- `Organizations`와 일부 `Places`
- `CreativeWorks`와 일부 `People`
- `People`과 일부 `ScienceAndNature`
- noise 100건 중 72건이 `People`

즉 leaf 단계의 높은 품질이 top 단계에서 완전히 보존되지는 않는다. 중심 cosine
하나만으로 상위 주제를 정의할 때 생기는 의미적 혼합을 추가로 줄여야 한다.

## 6. 기존 계층 FCM과의 현재 비교

동일한 Gemini 3,000건, seed 42의 기존 production fuzzifier 보고서와 비교한다.

| 방법 | top NMI | top ARI | leaf NMI | leaf ARI | runtime |
|---|---:|---:|---:|---:|---:|
| 계층 FCM, consensus auto m | 0.2037 | 0.1368 | 0.7196 | 0.4946 | 142.73초 |
| 계층 FCM, fast auto m | 0.2037 | 0.1368 | 0.7124 | 0.4799 | 92.27초 |
| HDBSCAN bottom-up | **0.7162** | **0.6493** | **0.8222** | **0.7122** | **36.84초** |

현재 측정에서 HDBSCAN bottom-up은 fast FCM보다 약 `2.5배`, consensus FCM보다
약 `3.9배` 빨랐다. HDBSCAN 자체는 UMAP-20 이후 약 0.1초 수준이며 실행시간의
대부분은 PCA와 UMAP이 차지한다.

이 표는 방향성 근거이지 최종 승자 판정은 아니다.

- FCM은 각 노드의 K를 자동 선택하지만 bottom-up의 6/17/18 cut은 정답 군집 수를
  알고 잘랐다.
- FCM은 완전한 재귀 소프트 계층과 증분 업데이트를 이미 제공한다.
- HDBSCAN bottom-up은 아직 seed 간 분할 안정성, 자동 레벨 선택과 증분 정책이
  완성되지 않았다.
- 두 보고서의 출력 구조와 비용 범위가 완전히 같지는 않다.

## 7. 자동 레벨 선택의 현재 한계

정답 라벨을 보지 않고 연속 병합 거리의 가장 큰 증가만 보면 상위 후보는 다음과
같았다.

| 순위 | cut 군집 수 | 병합 거리 증가 |
|---:|---:|---:|
| 1 | 23 | 0.1960 |
| 2 | 21 | 0.1249 |
| 3 | 8 | 0.0816 |
| 4 | 14 | 0.0744 |
| 5 | 18 | 0.0544 |

18개 cut은 구조적 후보로 나타났지만 정답 top인 6개와 middle인 17개는 강한
거리 gap으로 자동 선택되지 않았다. 라벨을 사용해 모든 cut을 평가하면 top은
6개에서 가장 좋지만, middle과 leaf는 병합 전 24개에서 NMI/ARI 평균이 가장
높다. 이는 다음을 의미한다.

1. 정답 분류 체계와 임베딩의 자연 밀도 구조가 완전히 같지는 않다.
2. 외부 라벨 점수만 최대화하면 과분할을 선호할 수 있다.
3. 운영 계층의 레벨 수는 merge gap 하나가 아니라 안정성·복잡도·최소 크기를
   함께 사용해야 한다.

## 8. 데이터 누수와 해석 제한

현재 수치는 탐색 결과다. 다음 과정에서 정답 정보를 사용했다.

- UMAP 차원과 거리, PCA 후 L2 여부를 같은 3,000건의 class NMI/ARI로 비교
- bottom-up 트리를 정답과 같은 6·17·18개에서 절단
- 각 정답 레벨에 가장 좋은 cut을 사후 탐색

HDBSCAN 학습과 병합 자체에는 정답 라벨을 넣지 않았지만, 파이프라인 선택과
평가가 같은 데이터에서 이뤄졌으므로 일반화 성능으로 해석하면 안 된다. 최소한
고정된 개발/평가 분할 또는 별도 데이터셋 검증이 필요하다.

## 9. 다음 연구 계획

### 단계 A: leaf 안정성 확정

1. seed `42, 43, 44, 45, 46`의 labels와 memberships를 모두 저장한다.
2. seed 쌍별 partition ARI/NMI와 noise Jaccard를 계산한다.
3. Hungarian matching으로 leaf 대응 관계와 중심 cosine 안정성을 측정한다.
4. 평균 외부 품질뿐 아니라 최악 seed를 기록한다.

완료 기준:

- 평균 noise `<= 5%`
- seed 간 partition ARI 평균 `>= 0.85`
- class NMI 평균 `>= 0.80`, ARI 평균 `>= 0.68`
- 3,000건 cold-run `<= 60초`

### 단계 B: 병합 거리 ablation

다음 상위 병합 거리를 같은 leaf에 적용한다.

1. 소프트 중심 cosine
2. 하드 core 중심 cosine
3. leaf 간 전체 문서 average cosine
4. 중심 cosine + 군집 내 분산 페널티
5. leaf membership 프로필의 Jensen-Shannon 거리

top/middle/leaf NMI·ARI, dendrogram purity, seed 간 tree cut ARI를 함께 평가한다.

### 단계 C: 비지도 cut 선택

정답 군집 수 없이 다음 기준을 비교한다.

- 병합 거리의 절대·상대 gap
- cut별 PCA 공간 silhouette
- bootstrap/seed 간 cut 안정성
- 최소·최대 부모 크기
- 트리 복잡도 페널티를 포함한 목적 함수

정답 6/17/18은 평가에만 사용하고 cut 선택 함수에는 전달하지 않는다.

### 단계 D: soft 계층 출력

leaf membership을 부모별로 합산해 다음을 저장한다.

- 레벨별 absolute membership
- 레벨별 conditional membership
- noise mass
- leaf→parent 경로
- 경로별 누적 소속도

기존 FCM assignments와 동일한 소비자가 읽을 수 있도록 공통 자료형을 정의한다.

### 단계 E: 증분 처리

초기 후보는 다음과 같다.

```text
신규 임베딩
  → 저장된 L2/PCA/UMAP transform
  → HDBSCAN approximate_predict
  → 기존 leaf→parent 경로 적용
  → noise·membership·중심 변화 누적
  → 임계값 초과 시 leaf와 bottom-up 트리 전체 재적합
```

HDBSCAN의 새 leaf 생성과 기존 leaf 분할은 `approximate_predict`만으로 처리할 수
없으므로, drift 감지와 전체 재적합 정책이 반드시 필요하다.

### 단계 F: 일반화 검증

- 개발 데이터에서 파라미터를 고정한 뒤 별도 평가 데이터에서 재측정
- 다른 임베딩 모델과 다른 문서 도메인에서 반복
- 정답 라벨이 없는 데이터에서는 사람이 병합 경로와 대표 문서를 평가
- 기존 FCM과 동일한 runtime/RSS/상태 크기/증분 시나리오로 비교

## 10. 현재 판단

HDBSCAN bottom-up은 현재까지 가장 유망한 속도·품질 절충 후보다.

- PCA 직접 HDBSCAN의 47.5% noise 문제를 UMAP-20 특징 공간으로 크게 완화했다.
- 24개 leaf에서 class NMI/ARI가 높고, 18개로 병합해도 손실이 작다.
- PCA 의미 공간의 bottom-up 병합이 정답 top/middle 계층과도 상당히 정렬된다.
- 3,000건 전체 계층 생성이 약 37초로 기존 재귀 FCM보다 빠르다.

그러나 자동 cut, seed 분할 안정성, 평가 데이터 분리, 증분 갱신이 남아 있다.
따라서 현재 결정은 **기본 알고리즘 교체가 아니라 독립 연구 트랙으로 확장**하는
것이다. 단계 A~C가 완료 기준을 통과하면 FCM과 동일 조건의 최종 대조 실험으로
진행한다.

## 11. 구현과 재현

구현:

- `hdbscan_bottom_up.py`: leaf 생성, 소프트 중심, weighted average linkage,
  tree cut, 평가와 결과 저장
- `test_hdbscan_bottom_up.py`: 병합·cut·소프트 중심·noise 보존 단위 테스트
- `requirements.txt`: `hdbscan>=0.8.40,<0.9`

실행:

```bash
./.venv/bin/python hdbscan_bottom_up.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --output-dir benchmarks/hdbscan-bottom-up-2026-08-15
```

결과:

- `benchmarks/hdbscan-bottom-up-2026-08-15/report.json`
- `benchmarks/hdbscan-bottom-up-2026-08-15/assignments.csv.gz`

`assignments.csv.gz`에는 원본 metadata와 함께 HDBSCAN leaf, probability,
outlier score, bottom-up `K=6/17/18/24` 할당이 저장된다. 현재 전체 테스트
113개가 통과한다.
