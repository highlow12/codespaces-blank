# 의미 임베딩 계층 클러스터링

문서와 원자적 노트를 수동 폴더 분류 없이 의미 기반으로 정리하기 위한 연구 프로젝트입니다. 하나의 문서가 여러 주제에 걸칠 수 있다는 전제에서 소프트 소속도, 계층 구조, 증분 업데이트를 함께 다룹니다.

> **현재 기본 경로:** 본문 임베딩 → 정규화 PCA → 계층형 Spherical Fuzzy C-Means(SFCM)
>
> **최신 연구 후보:** UMAP 공간에서 HDBSCAN leaf를 찾고, PCA 의미 공간에서 bottom-up으로 병합하는 계층화 방식입니다. 초기 결과는 유망하지만 아직 기본 알고리즘을 대체하지 않습니다.

## 프로젝트 목표

- 원자적 노트를 별도의 수동 청킹 없이 검색·클러스터링 단위로 사용합니다.
- 서로 의미가 겹치는 노트를 하나의 하드 라벨 대신 여러 군집의 소프트 소속도로 표현합니다.
- 군집화 공간과 2차원 시각화 공간을 분리해, 보기 좋은 투영이 군집 구조를 결정하지 않게 합니다.
- 새 문서가 들어올 때 전체 데이터를 항상 다시 계산하지 않고 상태를 안전하게 갱신합니다.
- 태그와 노트의 흐름은 본문 임베딩에 직접 더하지 않고 metadata, 링크, prior, reranking 같은 별도 채널로 보존합니다.

## 현재 상태

| 영역 | 상태 | 요약 |
| --- | --- | --- |
| 계층형 SFCM | 기본 경로 | PCA 차원, fuzzifier, 노드별 K를 데이터에 맞춰 선택하고 계층별 소프트 소속도를 생성합니다. |
| 증분 업데이트 | 구현 완료 | ID 기반 추가·교체, 선택적 membership 갱신, 드리프트 재클러스터링, 멱등 batch 처리, checksum·atomic save를 지원합니다. |
| HDBSCAN bottom-up | 연구 구현 완료 | 24개 leaf 발견과 PCA 공간 병합을 구현했습니다. 자동 cut, seed 안정성, 별도 평가 데이터, 증분 정책 검증이 남았습니다. |
| HDBSCAN noise 소프트 할당 | 비교 실험 | 근사 medoid와 최근접 문서 거리 기반의 사후 소속도를 비교합니다. 기본 클러스터링 경로는 아닙니다. |
| 원자적 노트 임베딩 비교 | 데이터 준비 | Gemini의 classification, retrieval_document, task type 미지정 임베딩 파일이 추가되어 있습니다. |
| 태그 융합 | 기본 경로에서 제외 | 합성 실험에서 태그 가중치를 줄일수록 좋아지는 조건이 확인되어 early fusion을 채택하지 않았습니다. |

## 알고리즘 구성

### 기본 운영 경로

~~~text
JSON 임베딩
  → 행별 L2 정규화
  → 클러스터링용 PCA 차원 자동 선택
  → 안정성 probe로 fuzzifier 선택
  → 재귀 Spherical FCM
  → multi-metric 또는 표본 합의 기반 K 선택
  → core / boundary / noise 판정
  → 계층별 지역 소속도와 경로 소속도
  → 별도의 시각화 PCA + 약지도 UMAP-2
  → 모델·중심·좌표·증분 통계를 상태 파일로 저장
~~~

기준 진입점은 **incremental_clustering.py**의 fit/update입니다. 구현과 기본값의 상세 명세는 [현재 알고리즘 문서](.mds/CURRENT_ALGORITHM.md)를 참조하세요.

### HDBSCAN bottom-up 연구 경로

~~~text
원본 임베딩
  → L2 정규화
  → PCA-96
  → PCA 결과 L2 정규화
  → UMAP-20
  → HDBSCAN으로 flat leaf 발견
  → PCA 공간에서 membership 가중 leaf 중심 계산
  → 중심 cosine 거리의 질량 가중 average linkage
  → 원하는 K에서 트리 절단
~~~

HDBSCAN은 leaf 발견에만 사용하고 상위 계층은 PCA 의미 공간에서 만듭니다. 구현은 **hdbscan_bottom_up.py**, 연구 과정과 한계는 [HDBSCAN bottom-up 연구 문서](.mds/HDBSCAN_BOTTOM_UP_RESEARCH.md)에 있습니다.

## 데이터

| 파일 | 용도 |
| --- | --- |
| **dbpedia_gemini_embeddings.json.gz** | 현재 주 검증 데이터. DBpedia 문서 3,000건, Gemini 임베딩 3,072차원입니다. |
| **dbpedia_label_embeddings.json** | 태그 라벨 18개의 임베딩입니다. 클러스터링 품질·성능 검증에 사용하지 않습니다. |
| **notes_gemini_classification_embeddings.json** | 원자적 노트의 classification task type 임베딩입니다. |
| **notes_gemini_retrieval_document_embeddings.json** | 같은 노트의 retrieval_document task type 임베딩입니다. |
| **notes_gemini_task_type_unspecified_embeddings.json** | 같은 노트의 task type 미지정 임베딩입니다. |

DBpedia는 재현 가능한 기준선으로 유지합니다. 다음 일반화 평가는 더 어렵고 주제 중첩이 많은 위키형 문서 데이터로 확장할 예정입니다.

## 설치

Python 가상환경을 만들고 의존성을 설치합니다.

~~~bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
~~~

## 실행

### 빠른 기본 적합

Gemini DBpedia 데이터 100건 표본으로 클러스터링 경로를 빠르게 확인합니다.

~~~bash
./.venv/bin/python incremental_clustering.py fit   --input-json dbpedia_gemini_embeddings.json.gz   --dataset-sample-size 100   --dataset-sample-seed 42   --fast   --state-output /tmp/incremental-fast.state.pkl   --skip-visualization
~~~

시각화와 증분 업데이트에 사용할 최종 상태를 만들 때는 **--skip-visualization**을 제거하고 출력 경로를 지정합니다.

~~~bash
./.venv/bin/python incremental_clustering.py fit   --input-json dbpedia_gemini_embeddings.json.gz   --state-output results/model.state.pkl   --assignments-output results/model_assignments.csv   --coordinates-output results/model_coordinates.csv   --tree-output results/model_tree.json   --plot-output results/model_scatter.png
~~~

### 증분 업데이트

~~~bash
./.venv/bin/python incremental_clustering.py update   --state results/model.state.pkl   --input-json new_embeddings.json   --state-output results/model_updated.state.pkl   --assignments-output results/model_updated_assignments.csv   --coordinates-output results/model_updated_coordinates.csv
~~~

기존 ID는 교체되고 새 ID는 추가됩니다. 저장된 PCA, 계층 중심, UMAP 좌표계를 재사용하며 드리프트 조건을 넘을 때만 전체 재클러스터링합니다.

### HDBSCAN bottom-up 재현

~~~bash
./.venv/bin/python hdbscan_bottom_up.py   --input-json dbpedia_gemini_embeddings.json.gz   --output-dir benchmarks/hdbscan-bottom-up-2026-08-15
~~~

이 스크립트는 현재 평가용 metadata인 class와 3단계 class_hierarchy를 요구합니다. 결과는 report.json과 assignments.csv.gz에 저장됩니다.

### 기타 실험 진입점

| 파일 | 용도 |
| --- | --- |
| **full_pipeline.py** | flat Auto-PCA SFCM과 시각화 전체 파이프라인 |
| **hdbscan_soft_pipeline.py** | HDBSCAN noise의 medoid·최근접점 사후 소속도 비교 |
| **benchmark_incremental_updates.py** | fit, update, 선택 refresh, 상태 크기, peak RSS 측정 |
| **benchmark_production_m_fuzzifier.py** | 고정·자동 fuzzifier 경로 비교 |

## 검증 결과

### 계층형 SFCM 증분 기준선 — 2026-08-08

Gemini 3,000건, seed 42, **--fast**, 시각화 포함, update 크기 4% 조건입니다.

| 입력 문서 수 | fit | 일반 update | 선택 refresh | refresh / skip | 상태 파일 | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 26.5초 | 4.07초 | 2.32초 | 15 / 80 | 5.5 MB | 0.87 GB |
| 500 | 31.2초 | 4.34초 | 0.47초 | 86 / 404 | 25.2 MB | 0.87 GB |
| 1,000 | 39.2초 | 4.94초 | 0.37초 | 40 / 940 | 38.5 MB | 0.97 GB |
| 3,000 | 77.7초 | 5.43초 | 1.51초 | 120 / 2,820 | 88.4 MB | 1.36 GB |

### HDBSCAN bottom-up 탐색 결과 — 2026-08-15

Gemini 3,000건, seed 42, PCA-96, UMAP-20, min_cluster_size 40, min_samples 2 조건입니다. HDBSCAN이 24개 leaf를 찾았고 100건, 3.33%를 noise로 판정했습니다. 전체 실행 시간은 36.84초였습니다.

| 평가 레벨 | 사용한 cut K | NMI | ARI | noise |
| --- | ---: | ---: | ---: | ---: |
| top | 6 | 0.7162 | 0.6493 | 3.33% |
| middle | 17 | 0.7945 | 0.6421 | 3.33% |
| leaf | 18 | 0.8222 | 0.7122 | 3.33% |
| 병합 전 leaf | 24 | 0.8351 | 0.7477 | 3.33% |

상세 결과는 [benchmark report](benchmarks/hdbscan-bottom-up-2026-08-15/report.json)에 있습니다.

이 수치는 일반화 성능이나 최종 승자 판정이 아닙니다.

- 같은 3,000건으로 UMAP·HDBSCAN 설정을 탐색하고 평가했습니다.
- 6, 17, 18이라는 정답 군집 수를 알고 트리를 잘랐습니다.
- HDBSCAN 경로에는 아직 자동 계층 레벨 선택과 완성된 증분 업데이트가 없습니다.
- 기존 SFCM 비교와 출력 구조·비용 범위가 완전히 같지 않습니다.

### 자동 fuzzifier 검증 — 2026-08-15

Gemini 3,000건의 seed 42, 43, 44에서 일반 자동 경로와 fast 자동 경로는 모두 m=1.2를 선택했습니다. fast 자동 경로의 평균 실행 시간은 66.9초, 일반 자동 경로는 135.5초였습니다. 이는 해당 데이터셋의 비교 결과이며 다른 데이터에 대한 보증이 아닙니다. 자세한 값은 [production fuzzifier report](benchmarks/production-m-fuzzifier-2026-08-15/report.json)에 있습니다.

## 핵심 설계 결정

- 기본 경로는 **content → PCA → SFCM**이며 태그를 임베딩에 직접 합치지 않습니다.
- 원본 임베딩은 보존하고 PCA는 군집화와 시각화를 위한 파생 표현으로 취급합니다.
- UMAP은 군집을 만드는 기준이 아니라 결과 탐색과 시각화를 위한 별도 모델입니다.
- 서로 다른 부모의 지역 membership을 직접 비교하지 않고 부모까지의 확률을 곱한 경로 membership을 사용합니다.
- 노트의 순서·흐름은 클러스터 좌표에 강제로 넣기보다 원문 ID, metadata, 외부 링크로 보존합니다.
- HDBSCAN bottom-up은 현재 가장 유망한 대안이지만 검증 항목을 통과하기 전까지 독립 연구 트랙으로 유지합니다.

의사결정의 전체 타임라인은 [프로젝트 결정 기록](.mds/PROJECT_DECISION_HISTORY.md)에 있습니다.

## 다음 연구 순서

1. HDBSCAN seed 42, 43, 44, 45, 46의 partition 안정성, noise Jaccard, leaf 중심 안정성을 측정합니다.
2. 정답 군집 수 없이 merge gap, silhouette, bootstrap 안정성, 복잡도 페널티로 계층 cut을 선택합니다.
3. 개발·평가 데이터를 분리하고 더 어려운 위키형 데이터에서 DBpedia 결과가 일반화되는지 확인합니다.
4. HDBSCAN leaf membership을 부모로 합산한 공통 soft hierarchy 출력과 증분 drift 정책을 설계합니다.
5. 원자적 노트에서 Gemini embedding task type 세 가지가 검색·소프트 클러스터링에 미치는 영향을 비교합니다.
6. 태그는 early fusion보다 metadata prior와 reranking을 우선 비교합니다.

태그 실험의 세부 범위는 [합성 태그 융합 계획](SYNTHETIC_TAG_FUSION_EXPERIMENT_PLAN.md)과 [완료된 결과](benchmarks/synthetic-tag-fusion-2026-08-09/RESULTS.md)에 있습니다.

## 테스트

~~~bash
./.venv/bin/python -m unittest discover -s . -p 'test_*.py'
~~~

2026-08-15의 마지막 기록에서는 전체 테스트 113개가 통과했습니다.
