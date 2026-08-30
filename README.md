# DBpedia 임베딩 클러스터링

DBpedia 문서 임베딩을 대상으로 **PCA → UMAP → HDBSCAN** 기반 클러스터링을
기본 경로로 실험·운영하는 프로젝트입니다. 기존 SFCM은 계층형·증분 처리와
비교 실험을 위한 명시적 호환 경로로 유지합니다.

## 현재 상태

| 영역 | 상태 | 요약 |
| --- | --- | --- |
| 증분 계층형 클러스터링 | 완료 · 통합됨 | ID 기반 추가/교체, 선택적 membership 갱신, 드리프트 기반 재클러스터링을 지원합니다. |
| 상태 저장 및 재실행 안전성 | 완료 | batch ID 멱등성, checksum envelope, atomic save, 동시 update 직렬화, 이전 상태 호환성을 갖춥니다. |
| 성능 기준선 | 측정 완료 | 2026-08-08 기준 3,000건 Gemini 임베딩에서 fast 모드 fit 77.7초, 일반 update 5.43초를 기록했습니다. 최근 fuzzifier 비교는 아래 검증 결과를 참조합니다. |
| 태그 융합 | 운영 검증 전 | fixed `K=10` 합성 sweep으로 이득·손해 조건은 확인했지만, 실제 운영 태그와 K 미지의 계층 경로 검증이 남았습니다. 기본 경로에는 넣지 않았습니다. |

### 현재 결론

- 기본 설계는 `content → PCA → UMAP → HDBSCAN`입니다. UMAP 공간에서 HDBSCAN이
  데이터가 지지하는 군집 수와 noise를 결정하며, HDBSCAN native membership도
  함께 저장합니다.
- 계층형·증분 SFCM 호환 경로에서는 500건 이상 노드의 최초 K를 20% 독립 표본의
  3/5 다수결로 고른 뒤 선택된 K만 전체 데이터에서 적합합니다. 이는 기본
  HDBSCAN discovery 경로가 아니라 `full_pipeline.py`/`incremental_clustering.py`의
  동작입니다.
- 같은 SFCM 호환 경로에서 `--fuzzifier`를 생략하면 안정성 probe로 `m`을
  선택합니다. 기본 후보는 `1.2, 1.4, 1.6, 1.8, 2.0`입니다.
- 태그는 곧바로 임베딩에 합치지 않고, 검증 전까지 metadata·prior·reranking 후보 채널로 분리합니다.
- 증분 업데이트는 전체 문서를 매번 다시 계산하지 않습니다. 중심 이동의 영향이 충분한 문서와 새로 추가·수정된 문서만 membership을 갱신하고, 자연 noise 또는 XB 품질 저하가 감지될 때 전체 재클러스터링을 수행합니다.

### Wikipedia HDBSCAN 이웃 검색 결론

Wikipedia 720건 품질 검증에서 exact-kNN과 PyNNDescent ANN의 선택 leaf가
99% 이상 일치했지만, 100~10,000건 단일 실행의 인덱스 구축·메모리 비용까지
포함하면 exact-kNN이 더 빠르고 작았습니다. 따라서 현재 HDBSCAN 기본값은
`neighbor_backend="exact"`이며, PyNNDescent는 반복 query가 훨씬 많거나 더 큰
데이터에서 별도로 검증하는 실험용 옵션으로만 유지합니다. ANN을 사용하려면
`fit_discovery(..., neighbor_backend="pynndescent")` 또는 해당 벤치마크의 CLI
옵션을 명시적으로 지정해야 합니다.

Calibration은 discovery별 PCA·이웃 인덱스·UMAP을 한 번만 적합하고, 선택된
HDBSCAN state와 projection을 held-out test에 재사용합니다. 비교 벤치마크의
`run.json`에는 PCA, neighbor index/query, UMAP fit/transform, HDBSCAN calibration
및 selected-state reuse, FCM, 시각화의 wall time과 stage peak RSS가 각각 기록됩니다.

## 선택의 타임라인

프로젝트에서 지금까지 선택한 방향과 그 판단 근거입니다.

| 시점 | 선택 | 이유와 현재 반영 상태 |
| --- | --- | --- |
| 2026-08-02 | 하드·평면 군집 대신 **계층형 구면 FCM** 채택 | 문서는 여러 주제에 걸칠 수 있고 주제에는 상·하위 구조가 있으므로, 하나의 하드 라벨보다 소프트 소속도와 재귀 분할을 보존하기로 했습니다. 입력과 중심을 단위 구면에 두어 의미적 방향을 기준으로 계산합니다. |
| 2026-08-02 | 군집화와 2차원 시각화를 분리 | UMAP에서 잘 갈라져 보인다고 해서 의미 군집이 좋은 것은 아니므로, 군집은 고차원 PCA 공간에서 수행하고 UMAP은 결과를 보여 주는 역할로 한정했습니다. |
| 2026-08-03 | 지역 소속도를 조건부 경로 소속도로 저장 | 서로 다른 부모 노드의 소속도를 직접 비교하지 않고, 부모까지의 확률을 곱한 경로 확률로 비교 가능하게 만들었습니다. 현재 이 상세 출력은 필요한 경우에만 opt-in합니다. |
| 2026-08-03 | 시각화에 소프트 소속도를 약하게만 반영 | 군집 라벨이 원래 임베딩 구조를 덮어쓰지 않도록 UMAP의 약지도 목표 가중치를 `0.01`로 낮게 유지했습니다. |
| 2026-08-03~04 | PCA 차원을 작업별로 분리한 뒤 자동 선택으로 전환 | 과거 고정 PCA-256(군집)·PCA-64(시각화) 비교를 바탕으로, 현재는 데이터별 k-NN 보존율이 포화되기 전의 차원을 각각 자동 선택합니다. |
| 2026-08-04 | 고정 K 대신 노드별 `multi_metric` K 선택 | 계층의 모든 노드를 같은 수로 나누지 않고, XB·실루엣·재시작 안정성·분할 계수를 함께 평가해 데이터가 지지하는 경우에만 분할합니다. |
| 2026-08-04 | 경계·이상치 문서를 별도 판정하고, 품질 악화 때만 재클러스터링 | 낮은 소속도·중심 거리·XB 악화를 함께 사용해 무조건적인 재학습과 과도한 분할을 피하도록 했습니다. |
| 2026-08-05 | 실험과 빠른 반복을 위해 `--fast` 경로와 재현 가능한 표본 추출 추가 | K 탐색용 표본, 적응적 fuzzifier, 제한된 refinement를 사용하되, 일반 선택기는 유지해 정확도와 탐색 속도를 구분했습니다. |
| 2026-08-05 | 실제 검증 데이터로 18개 태그 라벨 파일 대신 3,000건 Gemini 임베딩 사용 | 작은 태그 전용 파일은 클러스터링 품질·증분 처리 성능을 검증할 데이터가 아니므로, 표본 검증도 Gemini 데이터에서 수행하기로 했습니다. |
| 2026-08-05 | 정답 class에서 만든 태그 결합 결과를 운영 근거로 쓰지 않음 | 초기 태그 실험은 태그 신호의 상한을 확인했지만, 실제 운영 태그가 아니었습니다. 따라서 태그를 기본 경로에 편입하지 않고 별도 검증 대상으로 남겼습니다. |
| 2026-08-05 | 증분 상태를 문서별 outer product 대신 compact weight로 저장 | 상태 크기와 일반 update 비용을 낮추기 위해서입니다. 변경된 ID의 contribution만 빼고 더하는 delta update를 기본으로 채택했습니다. |
| 2026-08-06 | 전체 membership refresh 대신 중심 영향 기반의 선택 refresh | 중심이 충분히 움직였고 fuzzy weight가 큰 문서, 그리고 신규·수정 문서만 다시 계산하도록 선택했습니다. 전체 갱신은 실제 재클러스터링 때만 수행합니다. |
| 2026-08-06 | 즉시 noise 반응 대신 누적·EWMA·hysteresis·cooldown 기반 드리프트 판정 | 작은 배치의 우연한 noise로 재클러스터링이 반복되는 것을 막기 위해서입니다. |
| 2026-08-07 | flat/계층형 경로의 batch 처리 규칙을 공통 코어로 통합 | append/replace, batch ID 멱등성, replay 기록, atomic save를 동일한 방식으로 보장하기 위해서입니다. |
| 2026-08-08~09 | 수학적으로 동등한 계산 재사용과 혼합 정밀도 저장을 우선 | FCM 거리·membership 계산과 PCA 탐색의 중복을 줄이고, 임베딩·상태는 기본 `float32`, 중심·품질 통계는 `float64`로 유지해 결과를 보존하면서 CPU·RSS를 낮췄습니다. |
| 2026-08-09 | Python worker 병렬화는 보류 | 2 CPU 환경의 측정에서 순차 실행보다 느렸습니다. 따라서 복잡한 병렬화보다 입력 cache·lazy import·수치 계산 재사용을 우선합니다. |
| 2026-08-13 | 큰 노드의 기본 최초 K 선택을 표본 합의 방식으로 전환 | 20% 표본의 단일 선택은 불안정했지만 5개 독립 표본 중 3표 다수결은 3,000건 검증 조건 9/9에서 전체 선택 K와 일치했습니다. 선택 K는 전체 데이터에서 다시 적합하며, 합의 실패 시 exact 탐색으로 복귀합니다. |
| 2026-08-14~15 | 안정성 기반 fuzzifier 선택을 기본화하고 geometry·운영 경로를 재검증 | 기본 경로는 `m=2.0` 고정이 아니라 안정적인 후보를 선택합니다. 3,000건 Gemini, seed 42~44 비교에서 자동 경로는 모두 `m=1.2`를 선택했고, fast 자동 경로는 일반 자동 경로보다 평균 약 2.03배 빨랐습니다. 이 결과는 해당 데이터셋의 비교 근거이며 모든 데이터에 대한 보증은 아닙니다. |
| 현재 | 태그의 early fusion을 기본 경로에서 제외하고 `content → PCA → UMAP → HDBSCAN` 채택 | 태그 신호 자체는 확인하되, 현재 품질에서는 본문 공간의 기하를 해칠 가능성이 있습니다. SFCM은 계층형·증분 및 비교 실험 경로로 유지합니다. |

## 최근 검증 및 기준선

Gemini 임베딩 데이터셋(3,000건, 3,072차원), seed `42`, `--fast`, 시각화 포함 조건에서 측정한 값입니다. update 크기는 입력의 4%입니다.

| 입력 문서 수 | fit | 일반 update | 선택 refresh | refresh / skip | 상태 파일 | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 26.5초 | 4.07초 | 2.32초 | 15 / 80 | 5.5 MB | 0.87 GB |
| 500 | 31.2초 | 4.34초 | 0.47초 | 86 / 404 | 25.2 MB | 0.87 GB |
| 1,000 | 39.2초 | 4.94초 | 0.37초 | 40 / 940 | 38.5 MB | 0.97 GB |
| 3,000 | 77.7초 | 5.43초 | 1.51초 | 120 / 2,820 | 88.4 MB | 1.36 GB |

증분 리팩터 통합 뒤 루트 테스트 57개와 gzip 입력 100건 fast-fit smoke test를 통과했습니다.

### 최신 fuzzifier 검증 (2026-08-15)

Gemini 3,000건에서 seed `42, 43, 44`, 시각화 제외 조건으로 고정 `m=2.0`,
고정 `m=1.2`, 일반 자동 `m`, fast 자동 `m`을 비교했습니다. 일반 자동 경로는
세 seed 모두 `m=1.2`를 골랐고, fast 자동 경로는 평균 실행 시간 `66.9초`,
일반 자동 경로는 `135.5초`였습니다. 외부 top/leaf NMI·ARI의 비가중 평균은
각각 `0.3848`, `0.3772`였습니다. 자세한 조건과 seed별 결과는
[`production-m-fuzzifier` 보고서](benchmarks/production-m-fuzzifier-2026-08-15/report.json)에
있습니다.

## 빠른 실행

의존성을 설치합니다.

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Gemini 데이터에서 빠르게 초기 상태를 적합합니다. `dbpedia_label_embeddings.json`은 태그 라벨 18개만 가진 파일이므로 클러스터링 검증에 사용하지 않습니다.

```bash
./.venv/bin/python incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --dataset-sample-seed 42 \
  --fast \
  --state-output /tmp/incremental-fast.state.pkl \
  --skip-visualization
```

기본 PCA + UMAP + HDBSCAN 파이프라인을 실행하려면 다음을 사용합니다.

```bash
./.venv/bin/python 실험파일.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --output-dir results
```

이 기본 실행은 HDBSCAN이 발견한 leaf를 membership-weighted soft center로
만든 뒤, PCA 공간의 cosine distance와 mass-weighted average linkage로
bottom-up 계층을 함께 생성합니다. `results/hdbscan_hierarchy_assignments.csv`에는
원래 leaf와 모든 dendrogram cut(`bottom_up_k*`)이, `results/hdbscan_hierarchy_tree.json`에는
merge 순서·거리·질량이 저장됩니다. 발견된 leaf가 0개 또는 1개인 경우에도
빈 merge 또는 단일 leaf 계층으로 안전하게 기록됩니다. 이 계층은 레거시
재귀 PCA+FCM 경로와 별개의 결과이며, 후자가 필요할 때만 `--hierarchical`을
명시합니다.

기존 SFCM flat 경로가 필요하면 `--pipeline 2_auto_pca_fcm`을 명시합니다.
계층형·증분 SFCM은 `full_pipeline.py`와 `incremental_clustering.py`에서
별도로 사용할 수 있습니다.

원래 HDBSCAN noise 문서에 대해 대표점(근사 메도이드) 거리와 클러스터 내 최근접
5개 문서 거리의 소프트 소속도를 비교하려면 다음을 사용합니다.

```bash
./.venv/bin/python hdbscan_soft_pipeline.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --dataset-sample-seed 42 \
  --pca-components 8 \
  --min-cluster-size 5 \
  --min-samples 3 \
  --output-dir results/hdbscan_soft
```

`assignments.csv`에는 원래 HDBSCAN 라벨과 대표점 기준 최종 `cluster`, 두 방식의
추천 라벨·최대 소속도·각 membership 열이 기록됩니다. 대표점 방식만 최종 라벨에
반영하며, 기본 임계값 `0.60` 미만은 noise(`-1`)로 유지합니다. 이 값들은 HDBSCAN이
제공하는 자체 확률이 아니라, PCA·L2 정규화 공간의 코사인 거리로 계산한 사후 비교
지표입니다.

### HDBSCAN 소프트 소속도 비교 실행 기록 (2026-08-15)

위 명령의 Gemini 데이터셋 100건 표본(`seed=42`, PCA 8차원, `min_cluster_size=5`,
`min_samples=3`)을 한 번 실행한 결과는 다음과 같습니다.

| 항목 | 결과 |
| --- | ---: |
| HDBSCAN hard cluster 수 | 9 |
| 원래 noise 문서 수 | 12 |
| 대표점 방식 재배정 수 (`>= 0.60`) | 0 |
| 최근접점 방식 재배정 수 (`>= 0.60`) | 0 |
| noise 추천 라벨 일치율 | 50.0% |
| 평균 confidence 차이 (대표점 − 최근접점) | 0.0555 |
| 평균 절대 confidence 차이 | 0.0611 |

이 작은 표본과 기본 임계값에서는 두 방법 모두 원래 noise를 보수적으로 유지했습니다.
이는 품질 일반화 결과가 아니라 파이프라인이 실제 데이터에서 동작하는지 확인한
스모크 실행 기록입니다. 세부 문서별 결과와 클러스터별 메도이드는
[`assignments.csv`](results/hdbscan_soft/assignments.csv)와
[`summary.json`](results/hdbscan_soft/summary.json)에 저장되어 있습니다.

증분 처리 비용을 측정하려면 다음을 사용합니다.

```bash
./.venv/bin/python benchmark_incremental_updates.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --update-size 10 \
  --fast \
  --output-json /tmp/incremental-benchmark.json
```

### Discovery와 membership 분리 비교 (Phase 1–3)

최종 파이프라인의 첫 검증 단계는 `L2 normalize → PCA → UMAP 20D → HDBSCAN leaf`
발견 결과를 두 방식으로 비교합니다. Native 방식은
`all_points_membership_vectors()`를 그대로 저장하고, Proposed 방식은 post-PCA
정규화를 하지 않은 PCA semantic space에서 고정 exact-kNN과 confidence-weighted
label propagation을 사용합니다. PCA 차원은 아래 명령처럼 `--pca-components`를
생략하면 기존 kNN preservation 기반으로 자동 선택됩니다.

```bash
./.venv/bin/python hdbscan_membership_comparison_pipeline.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --output-dir results/hdbscan_membership_comparison \
  --dataset-sample-size 100 \
  --dataset-sample-seed 42 \
  --umap-n-neighbors 8 \
  --min-cluster-size 5 \
  --min-samples 3 \
  --neighbor-count 8 \
  --seed 42
```

`assignments.csv`, `boundary_cases.csv`, `summary.json`에 두 방법의 membership,
unexplained mass, 비교 통계와 사람 검토 후보를 저장합니다. 현재 구현은 Phase 1–3
검증 범위이며 adaptive neighborhood, HNSW, semantic hierarchy, incremental
insertion은 아직 포함하지 않습니다.

## 다음 연구 작업

태그 융합은 아직 제품 경로에 반영하지 않았습니다. fixed `K=10` 합성 sweep에서는
태그 corruption·content noise·weight에 따라 early fusion의 이득과 손해가 전환됨을
확인했습니다. 다음 단계는 해당 결론을 운영 조건에 맞게 좁히는 일입니다.

1. K를 모르는 계층 경로에서 content-only·observed/oracle/shuffled tag와 fusion ablation을 다시 평가합니다.
2. 정답 `class`에서 만든 태그 대신 실제 수집 태그의 누락·오분류·구조적 오류를 모델링합니다.
3. early fusion과 metadata prior·reranking을 같은 soft membership 및 경계 문서 지표로 비교합니다.

세부 가설과 완료 범위는 [합성 데이터 기반 태그 융합 실험 계획서](SYNTHETIC_TAG_FUSION_EXPERIMENT_PLAN.md),
완료된 sweep의 수치는 [결과 문서](benchmarks/synthetic-tag-fusion-2026-08-09/RESULTS.md)에 정리되어 있습니다.

## 테스트

```bash
./.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```
