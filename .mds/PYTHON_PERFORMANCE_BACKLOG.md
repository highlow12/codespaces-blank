# Python 성능 최적화 Backlog 및 작업 트래커

이 문서는 `JS_PERFORMANCE_OPTIMIZATION_PLAN.md`에 아직 명시되지 않은 Python
실행 경로의 최적화 후보를 기록하고 추적한다. 알고리즘 결과나 출력 계약을
바꾸는 작업은 반드시 Gemini 데이터의 품질·수치 회귀 검증을 먼저 통과해야 한다.

## 발견 근거

2026-08-08에 다음 명령으로 Gemini 임베딩 3,000건에서 300건을 고정 표본화한
`--fast --skip-visualization` fit을 cProfile로 측정했다.

```bash
./.venv/bin/python -m cProfile -o /tmp/perf-optimization-current.prof \
  incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 300 \
  --dataset-sample-seed 42 \
  --state-output /tmp/perf-optimization-current.state.pkl \
  --pca-components 64 \
  --max-depth 3 \
  --min-node-size 30 \
  --min-child-size 10 \
  --max-clusters 4 \
  --fast \
  --skip-visualization
```

profile 전체는 48.50초였다. 이 값에는 cold import와 입력 decode가 포함되어
있으므로 FCM kernel의 warm 성능과 혼동하지 않는다.

| 구간 | 누적 시간 | 관찰 |
|---|---:|---|
| `load_embeddings_from_json` | 20.64초 | 300건 표본에도 gzip JSON 3,000건 전체를 decode |
| `json.decoder.raw_decode` | 14.02초 | Python 객체 materialization이 지배적 |
| 시각화 import 경로 | 8.50초 | `--skip-visualization`이어도 UMAP/plotting 모듈을 import |
| `spherical_fcm` | 14.40초 | 51회 FCM 실행 |
| `_minimum_center_distance` | 9.55초 | 반복 중 sklearn Euclidean 거리 호출 |
| `_scout_m` | 2.07초 | fast hierarchy의 노드별 fuzzifier probe |

`_minimum_center_distance`의 범용 거리 호출 제거는 기존 계획 8.2에 이미
포함되어 있으므로, 이 backlog에서는 새 항목으로 중복 등록하지 않는다. 다만
warm FCM 최적화를 시작할 때 가장 먼저 완료해야 하는 기존 작업이다.

### E-00 결과

- `_minimum_center_distance`를 단위 구면 중심의 `2 - 2 * dot(center_i, center_j)`
  제곱거리 계산으로 변경했다.
- 기존 normalized-center Euclidean 결과와 비교하는 회귀 테스트를 추가했다.
- 동일한 300건 Gemini profile에서 helper 누적 시간이 기존 9.55초에서 0.175초로
  줄었고, helper 내부의 sklearn `euclidean_distances` 호출은 제거됐다.
- 전체 테스트 65개가 통과했다.

## 새 최적화 후보

### P0-1. 시각화 의존성 lazy import

`incremental_clustering.py`는 시작 시 `cluster_visualization`을 import하며,
이는 plotting과 UMAP 의존성을 함께 불러온다. `--skip-visualization`은
`fit_visualization` 호출만 생략할 뿐 import 비용은 피하지 못한다.

시각화 함수와 타입 의존성을 `_fit_visualization` 및 실제 plot 생성 경로로 옮긴다.
클러스터링·update CLI는 UMAP/Matplotlib 없이 import 가능해야 한다.

### N-01 결과

- 시각화 기본값을 `visualization_constants.py`로 분리하고, `cluster_visualization`의
  함수 import를 fit/update/plot 실행 경로로 이동했다.
- `incremental_clustering` 초기 import에서 `cluster_visualization`,
  `cluster_plotting`, `umap_projection`이 로드되지 않는 회귀 테스트를 추가했다.
- Gemini 100건 고정 표본의 `--fast --skip-visualization` 실행에서 state가 정상 생성되고,
  전체 테스트 66개가 통과했다.

### P0-2. Python용 바이너리 입력 캐시와 부분 로더

현재 JSON loader는 표본 실행 전에도 전체 gzip JSON을 Python dict/list와
`float64` 배열로 materialize한다. JS 이식 계획의 binary format과 별도로, 현재
Python CLI가 직접 쓸 수 있는 `float32` row-major data file, manifest, metadata
형식을 도입한다.

`memmap` 또는 row-slice 가능한 array format으로 고정 seed 표본·update batch가
필요한 행만 읽도록 한다. JSON loader는 호환 fallback으로 유지하며, 변환 결과에는
입력 hash·shape·dtype·metadata schema version을 기록한다.

### N-02 결과

- `embedding_cache.py build`가 JSON/gzip JSON을 스트리밍해 `embeddings.f32`,
  `metadata.jsonl`, row offset index, manifest를 생성한다. manifest에는 source SHA-256,
  shape `[3000, 3072]`, `<f4` dtype, metadata schema version을 기록한다.
- `incremental_clustering.py`의 fit/update CLI에 `--input-cache`를 추가했다. 고정 seed
  표본은 memmap에서 필요한 행만 읽고 source JSON을 다시 decode하지 않는다.
- Gemini 3,000건 cache를 만든 뒤 seed=42 100건을 원본 JSON loader와 비교해 ID, metadata,
  float32 embedding을 일치시켰고, cache 기반 incremental fit의 clustering 결과도
  일치했다.
- 동일 프로세스 import 이후 측정한 load+sample baseline은 JSON 14.780초/peak RSS
  808,580KB, cache 0.009초/194,032KB였다. cache build는 9.470초, embedding data는
  36MB였다.
- 전체 테스트 69개가 통과했다.

### P1-1. fuzzifier scout 재사용

fast path는 hierarchy의 여러 노드에서 동일한 `m_values` probe를 반복한다.
루트에서 선택한 m을 자식 노드 또는 동일 depth의 노드가 우선 재사용하고,
restart stability가 임계값 미만일 때만 local re-scout하는 정책을 실험한다.

이 변경은 K 선택과 노이즈 품질에 영향을 줄 수 있으므로 exact 대비 선택 K,
ARI/NMI, noise 비율, m scout 호출 수를 함께 기록한다.

### N-03 결과

- fast hierarchy가 부모 node의 안정적인 m을 자식 node에 전달하고, K scout의
  restart stability가 `minimum_probe_stability` 미만일 때만 local m probe를 다시
  실행하도록 구현했다. 기본값은 활성화이며 `--no-fast-m-reuse`로 기존 정책을
  재현할 수 있다.
- Gemini 300건 표본 3개에서 m scout 호출은 `9/9/10회`에서 `1/2/1회`로 줄었고,
  hierarchy runtime 평균은 `2.129초`에서 `1.880초`로 11.7% 단축됐다.
- seed 43/44에서는 reuse 전후 hierarchy ARI/NMI가 동일했다. seed 42에서는
  no-reuse 대비 ARI가 `0.3065 → 0.3808`, NMI가 `0.6245 → 0.6143`이었다.
- 별도 Gemini exact-vs-fast K benchmark(100·300건, seed 42·43, 두 refine 설정)는
  8/8회 K가 일치했고, 300건의 평균 label ARI/NMI는 모두 1.0이었다.
- Gemini 1,000건 표본(seed 42, cache 입력)에서는 end-to-end CLI runtime이
  `10.835초 → 10.664초`로 1.6% 단축됐고, hierarchy fit runtime은
  `7.817초 → 7.578초`로 3.1% 단축됐다. local m probe는 20회에서 11회로
  줄었으며, 두 실행 모두 leaf 23개·noise 0건이고 1,000개 샘플의 cluster path가
  전부 일치했다.
- 전체 테스트 70개가 통과했다.

### N-04 결과

- 2 CPU/OpenBLAS 2-thread 환경에서 restart와 root sibling selector의
  worker 2 프로토타입을 측정했다. 병렬 경로는 worker당 BLAS thread를 1개로
  제한해 oversubscription을 피했다.
- Gemini cache 1,000건(seed 42)의 warm hierarchy fit 3회에서 sequential
  p50은 `8.092초`, worker 2는 `8.926초`로 10.3% 느려졌다. 전체 cluster path와
  noise 판정은 일치했다.
- root의 4개 sibling selector만 병렬화한 별도 측정에서도 p50이
  `0.800초 → 0.936초`로 17.1% 느려졌고, 선택 K/m은 일치했다.
- 현재 환경과 1,000건 규모에서는 executor 및 BLAS thread 제어 비용이 이득보다
  컸으므로 production 병렬화는 채택하지 않고 보류한다. 더 많은 CPU 또는 더 큰
  node가 실제 운영 조건이 될 때 재측정한다.

### N-05 결과

- `--embedding-storage-dtype {float32,float64}`를 추가하고 기본값을 `float32`로
  설정했다. 입력·PCA projection·state embedding은 저장 dtype을 따르며, FCM
  중심·membership·목적 함수·XB 계산은 `float64`를 유지한다. 기존에 dtype 설정이
  없는 legacy `float64` state는 update 시 `float64`를 유지한다.
- Gemini cache 100·300·1,000건, seed 42·43의 reference `float64`/`float32`
  비교 6회에서 cluster path·noise·선택 K가 모두 일치했다. 중심 최대 절대 차이는
  `3.94e-05`, XB 최대 상대 차이는 `1.22e-05`였다. update fixture에서도 path와
  noise가 일치했고 중심 최대 절대 차이는 `1.01e-08`, XB는 동일했다.
- 1,000건 cache, seed 42의 별도 프로세스 측정에서 peak RSS는
  `415,336KB → 308,184KB`로 25.8% 감소했고, state embedding bytes는
  `24,576,000 → 12,288,000`으로 50.0% 감소했다. pickle state 크기도
  `39,254,489 → 20,023,080 bytes`로 49.0% 줄었다. 같은 측정의 fit runtime은
  `8.089초 → 7.022초`였으며, 100·300건에서는 실행 편차로 속도 개선이 중립적이거나
  작았고 1,000건 직접 비교에서는 12.3~14.9% 단축됐다.
- 수치 허용오차, update/save/load dtype 보존, 기본 설정 회귀 테스트를 포함해 전체
  테스트 73개가 통과했다.

### P1-2. Python의 독립 FCM 작업 병렬화

현재 candidate K와 restart는 직렬 수행된다. Python 경로에도 독립 restart 또는
부모 split 이후의 형제 노드를 병렬화하는 실험 경로를 만든다.

난수 seed는 현재 규칙을 유지하고 결과를 고정 순서로 수집한다. NumPy/BLAS
자체의 thread와 중첩되지 않도록 worker 수와 BLAS thread 수를 함께 제어해야 한다.
K 후보는 XB 조기 종료 조건이 있으므로 무조건 병렬화하지 않고, restart 또는
이미 확정된 sibling 작업부터 대상으로 삼는다.

### P1-3. Python Float32 저장·계산 경로

입력 loader, embedding validation, PCA projection, state가 대부분 `float64`로
고정된다. embedding/PCA projection/상태 저장은 `float32`를 기본 후보로 두고,
중심 충분통계량·목적 함수·XB 누산만 `float64`를 유지하는 혼합 정밀도 경로를
검증한다.

이 작업은 메모리·I/O 대역폭을 줄이는 것이 목적이며, labels, memberships, 중심,
XB, incremental update 결과가 정한 허용오차를 만족할 때만 채택한다.

### P2-1. conditional path membership 생성 선택화

계층 fit은 모든 node에 대해 모든 document의 conditional path membership을
계산하고 assignment의 열로 저장한다. 이는 `O(rows × nodes × k)` 시간과
`O(rows × paths)` 출력 크기를 갖는다.

분석용 출력이 필요할 때만 생성하는 `include_conditional_memberships` 옵션을
검토한다. 기본값 또는 출력 schema를 바꾸기 전에는 downstream 소비자가 모든 path
membership 열을 요구하는지 확인해야 한다.

### N-06 결과

- fit API와 CLI에 `include_conditional_memberships`를 연결하고,
  `--include-conditional-memberships`로 opt-in할 수 있게 했다. 기본값은 `false`이며
  state config에 저장되어 update·membership refresh·recluster도 같은 assignment
  schema를 유지한다. 비활성화 시 conditional membership 계산과 path 열 생성을 모두
  건너뛴다.
- 조사 중 conditional membership helper의 누락된 반환을 복구하고,
  `visual_assignments.py`의 level/path 열 정규식 이중 escape를 수정했다. opt-in
  assignment는 path membership을 시각화 supervision의 soft target으로 사용하고,
  비활성화 assignment는 cluster label fallback으로 동작한다.
- Gemini cache seed=42에서 1,000건은 path 열이 `0개 → 40개`, state pickle이
  `20,023,048 → 20,346,392 bytes`로 1.6% 증가했다. 3,000건은 path 열이
  `0개 → 62개`, state pickle이 `47,213,936 → 48,707,303 bytes`로 3.1% 증가했다.
  peak RSS도 1,000건에서 `310,188KB/310,680KB`, 3,000건에서
  `584,832KB/586,916KB`(off/on)로 측정됐다. 반대로 기본 비활성화 경로는 이
  rows×paths 출력 비용을 제거한다.
- 3,000건 비교에서 cluster path·noise·level labels·tree가 일치했고, hierarchy
  center 최대 절대 차이는 `0`이었다. fit runtime은 1,000건에서 `6.992초/6.886초`,
  3,000건에서 `28.735초/28.407초`로 측정됐지만 실행 편차 범위이며, conditional
  계산은 전체 fit runtime의 지배 비용이 아니므로 안정적인 runtime 개선으로
  주장하지 않는다.
- opt-in/off assignment, update schema 보존, legacy state migration, 시각화 fallback
  및 기존 경로를 포함한 전체 테스트 79개가 통과했다.

### P2-2. 대형 상태의 envelope 직렬화 경로

checksum envelope는 payload를 bytes로 pickle한 뒤 envelope 자체를 다시 pickle해
atomic write한다. 300건 profile에서는 0.07초로 우선순위가 낮지만, 3,000건 state와
장기 update에서는 큰 bytes 복사·peak memory가 될 수 있다.

payload checksum을 stream 또는 sidecar manifest로 분리하고, checksum·legacy
load·atomic replace 계약을 유지하는지 대형 state에서 측정한다.

## CPU·RSS 공동 최적화 계획 (2026-08-09)

### R-00 결과 (2026-08-09)

`perf/optimization-plan`의 commit `294fdb5a3694072fbed9dbabe8279897b8b88246`에서
3,000건 Gemini gzip 원본으로 row-addressable cache를 만들고, cache 입력의
`fit --fast --skip-visualization`을 표본 크기별 새 프로세스 3회 실행했다. cache
생성 시간은 측정에서 제외했다.

- 원본: `dbpedia_gemini_embeddings.json.gz`, SHA-256
  `9a949bec1402b52f4b2cba4376ea3eda7c69003b33b7b1ea72e9501cf84d25fc`
- cache manifest SHA-256:
  `d27b00b5cee234330973727e999d82ef936cf13db6ed87768aba119867bb80d3`
- 입력 shape/dtype: `3000 x 3072`, `<f4`
- Python 3.12.1, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0
- 2 CPU 환경에서 `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`,
  `MKL_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`를 고정했고 seed와 표본 seed는
  모두 `42`로 고정했다.

실행 명령은 다음과 같다.

```bash
./.venv/bin/python incremental_clustering.py fit \
  --input-cache <gemini-cache> \
  --dataset-sample-size <1000-or-3000> \
  --dataset-sample-seed 42 \
  --seed 42 \
  --fast --skip-visualization \
  --state-output <state-path>
```

| 표본 | wall p50 (초) | 개별 wall 범위 (초) | peak RSS p50 (KiB) | state p50 (bytes) | PCA | root K | leaf/noise |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 9.202636 | 9.107045–9.427152 | 313,640 | 20,023,121 | 160 | 4 | 23 / 0 |
| 3,000 | 30.501021 | 30.290327–31.710225 | 592,508 | 47,214,009 | 160 | 4 | 36 / 0 |

wall 편차는 1,000건 3.48%, 3,000건 4.66%로 계획의 5% 중단 기준 이내다. 3,000건
root 선택값은 silhouette `0.0421347469`, Xie–Beni `0.2311225618`, selection
score `0.7`이며, 전체 hierarchy의 natural/forced/projection noise는 모두 0이다.

별도 `cProfile`에서도 두 표본의 hot path 순서가 일치했다.

| 표본 | hierarchy FCM | PCA 차원 선택 | fast FCM 후보 선택 | spherical FCM |
|---:|---:|---:|---:|---:|
| 1,000 | 7.758초 | 2.351초 | 5.156초 | 4.548초 |
| 3,000 | 28.924초 | 14.915초 | 13.310초 | 11.415초 |

3,000건 PCA 시간 중 `scipy.linalg.svd`가 7.344초, PCA prefix neighbor 평가가
6.318초였다. state envelope/checksum과 atomic pickle은 각각 약 0.038초(1,000건),
0.110초(3,000건)로 현재 fit CPU의 주 병목이 아니다. 3,000건 state에서 확인한 큰
수치 보유 구조는 입력 embedding 36,864,000 bytes, PCA components 6,291,456
bytes, assignments 1,744,064 bytes, metadata 767,130 bytes,
center contribution projected/weights 합계 2,158,536 bytes였다.

따라서 다음 구현 대상은 PCA 자동 차원 탐색의 projection·neighbor 계산 재사용인
**C-02**로 정한다. 현재 측정상 C-02가 fast FCM 후보 선택(C-01)보다 누적 시간이
크지만, PCA fit과 prefix 평가가 수학적으로 재사용 가능한지 먼저 확인한 뒤 하나의
변경으로 구현한다. 수치 결과·선택 차원·downstream assignment가 동일하지 않으면
채택하지 않는다.

### C-02 결과 (2026-08-09)

`pca_dimension_search.py`에 후보별 `k` 이웃 탐색을 최대 `k` 한 번으로 통합하는
`neighbor_indices_by_k`를 추가했다. reference embedding과 각 PCA prefix 모두에서
`k=15,30`을 따로 계산하던 경로를 `k=30` 검색 결과의 prefix view로 재사용한다.
이 변경으로 3,000건의 이웃 검색 호출은 34회에서 18회(전체 reference 1회와
17개 후보별 1회)로 줄었다.

| 표본 | baseline wall p50 (초) | C-02 wall p50 (초) | CPU 변화 | baseline RSS (KiB) | C-02 RSS (KiB) | state 변화 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 9.202636 | 8.492067 | -7.72% | 313,640 | 319,588 (+1.90%) | 0% |
| 3,000 | 30.501021 | 26.745578 | -12.31% | 592,508 | 592,720 (+0.04%) | 0% |

3,000건에서 CPU 10% 개선과 RSS 3% 이내 회귀 기준을 충족해 C-02를 채택한다.
1,000건 개선은 7.72%로 작지만 RSS 회귀는 3% 이내이고, 큰 운영 표본에서 목표를
넘겼다. 변경 전후 1,000/3,000건 state의 PCA 선택 차원, assignments, metadata,
embedding, hierarchy summary/root metric가 모두 같았고 hierarchy center 최대
절대 차이는 `0`이었다. PCA·clustering/visualization 관련 테스트와 루트 테스트
총 81개가 통과했다.

변경 후 3,000건 cProfile에서 `evaluate_pca_prefixes`는 6.318초에서 3.537초,
neighbor search는 34회·6.436초에서 17회·3.198초로 줄었다. 다음 병목은 fast FCM
후보 K·restart 경로인 **C-01**이다.

CPU 시간만 줄이는 변경이 peak RSS를 키우거나, 반대로 메모리만 줄이고 hot path를
느리게 만드는 것을 피하기 위해 이후 작업은 두 지표를 함께 측정한다. 작업별 범위,
설계 선택지, 결과 동등성 검증, 채택·중단 기준은
[PYTHON_CPU_RSS_OPTIMIZATION_PLAN.md](PYTHON_CPU_RSS_OPTIMIZATION_PLAN.md)에 상세히
정의한다.

| 우선순위 | 계획 ID | 작업 | 주 목표 | 상태 |
|---:|---|---|---|---|
| 0 | R-00 | Gemini cache warm CPU·RSS 기준선과 병목 프로파일 고정 | 이후 작업의 시간·RSS·state 기준값 확보 | 완료 |
| 1 | C-01 | FCM candidate K·restart 배열/계산 재사용 | CPU 우선, workspace 수명 관리로 RSS 제한 | 대기 |
| 2 | C-02 | PCA 자동 차원 탐색의 projection 재사용 | 반복 PCA/투영 CPU와 임시 배열 RSS 감소 | 완료 |
| 3 | M-01 | 증분 center contribution compact numeric 저장 | update CPU·state 크기·RSS 동시 감소 | 대기 |
| 4 | I-01 / N-07 | state envelope 직렬화 중복 제거 | 대형 state save/load CPU·일시 RSS 감소 | 보류: I/O profile 필요 |
| 5 | M-02 | level soft-membership의 필요 시 생성 추가 축소 | 소비되지 않는 출력의 RSS/state 감소 | 대기: field profile 필요 |
| 조건부 | H-01 / N-04 | worker 병렬화 재검토 | 충분한 CPU에서만 wall time 개선 검증 | 보류: 2 CPU에서 역효과 |

공통 채택선은 동일 Gemini cache·seed의 1,000/3,000 rows warm-run에서 CPU 시간 또는
peak RSS 10% 이상 개선, 그리고 다른 지표의 3% 초과 회귀가 없는 것이다. 작은 개선은
측정값과 보류 사유를 남기고 기본 경로에는 반영하지 않는다.

## 작업 트래커

| ID | 상태 | 작업 | 선행 조건 | 완료 기준 |
|---|---|---|---|---|
| E-00 | 완료 | 반복 중 `_minimum_center_distance`의 sklearn 거리 호출 제거 | JS 성능 계획 8.2 | 동일 centers/collapse 판정, warm FCM profile에서 범용 거리 호출 제거 |
| N-01 | 완료 | skip-visualization lazy import | 없음 | skip CLI가 UMAP/plotting을 import하지 않고 cluster 결과·state가 기존과 일치 |
| N-02 | 완료 | Python binary cache 및 부분 loader | N-01과 독립 | Gemini 표본 ID/embedding 일치, 전체 JSON materialization 없음, load 시간·RSS baseline 기록 |
| N-03 | 완료 | m scout 재사용 정책 | fast K benchmark | exact 대비 K/ARI/NMI/noise 기준 통과, scout 호출 수와 fit 시간 감소 |
| N-04 | 보류 | restart·sibling Python 병렬화 | thread/BLAS 제어 실험 | worker 1/N의 seed 결과 일치, oversubscription 없음, warm fit p50 개선 |
| N-05 | 완료 | Float32 Python 경로 | 수치 fixture 확장 | labels/중심/XB/update 허용오차 통과, input·state RSS 및 크기 감소 |
| N-06 | 완료 | conditional membership 선택화 | downstream schema 사용처 조사 | opt-in/off 계약 확정, 필요 없는 실행의 rows×paths 배열·열 미생성 |
| R-00 | 완료 | Gemini cache warm CPU·RSS 기준선 및 병목 프로파일 | 동일 환경·cache·seed 고정 | 1,000/3,000 rows p50·peak RSS·state 표와 CPU/RSS 상위 보유 구조 기록 |
| C-01 | 대기 | FCM candidate K·restart 배열/계산 재사용 | R-00에서 FCM 병목 확인 | 수치 결과 동등, CPU 또는 RSS 10% 개선, 다른 지표 3% 초과 회귀 없음 |
| C-02 | 완료 | PCA 자동 차원 탐색 projection 재사용 | R-00에서 PCA 병목 확인 | 선택 차원·downstream 결과 동등, CPU/RSS 공동 기준 통과 |
| M-01 | 대기 | 증분 center contribution compact numeric 저장 | contribution/state 필드별 크기 측정 | replace/idempotency·legacy load 통과, update CPU·state/RSS 개선 |
| N-07 / I-01 | 보류 | state envelope 복사·직렬화 개선 | 3,000건 이상 state I/O profile | checksum/legacy/atomic 계약 통과, peak RSS·save/load 시간 비교 |
| M-02 | 대기 | level soft-membership 필요 시 생성 추가 축소 | field-level 출력 메모리 profile | schema/visualization fallback 통과, 기본 경로 RSS/state 개선 |
| N-04 / H-01 | 보류 | restart·sibling Python 병렬화 재검토 | 4+ 실제 CPU와 큰 작업량 | seed 결과 일치, aggregate RSS 제한, warm fit p50 개선 |

## 향후 실행 순서

1. R-00으로 warm CPU·RSS 기준선과 실제 hot path를 확정한다.
2. C-01과 C-02 중 R-00에서 더 큰 CPU 병목으로 확인된 항목을 먼저, 나머지를 다음으로
   진행한다. 두 작업을 한 변경에 섞지 않는다.
3. M-01로 증분 update/state의 Python 객체 오버헤드를 줄인다.
4. I/O profile 또는 field profile 근거가 생기면 N-07/I-01, M-02를 진행한다.
5. N-04/H-01은 재개 조건을 충족할 때만 다시 실험한다.

## 공통 검증 규칙

- 클러스터링·증분 검증 데이터는 `dbpedia_gemini_embeddings.json.gz` 또는 원본
  JSON의 3,000건 Gemini 데이터만 사용한다.
- 빠른 검증은 `--dataset-sample-size`, 고정 seed, `--fast`를 사용한다.
- 알고리즘 변경은 exact 대비 K, ARI/NMI, noise 비율, XB와 중심 통계를 함께
  기록한다.
- 성능 결과에는 commit, 입력 hash, 설정, seed, cold/warm 구분, runtime, peak RSS를
  남긴다.
- 각 항목을 완료하면 이 문서의 상태와 기준선·결과 링크를 갱신한다.
