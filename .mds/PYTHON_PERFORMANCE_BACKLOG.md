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

### P0-2. Python용 바이너리 입력 캐시와 부분 로더

현재 JSON loader는 표본 실행 전에도 전체 gzip JSON을 Python dict/list와
`float64` 배열로 materialize한다. JS 이식 계획의 binary format과 별도로, 현재
Python CLI가 직접 쓸 수 있는 `float32` row-major data file, manifest, metadata
형식을 도입한다.

`memmap` 또는 row-slice 가능한 array format으로 고정 seed 표본·update batch가
필요한 행만 읽도록 한다. JSON loader는 호환 fallback으로 유지하며, 변환 결과에는
입력 hash·shape·dtype·metadata schema version을 기록한다.

### P1-1. fuzzifier scout 재사용

fast path는 hierarchy의 여러 노드에서 동일한 `m_values` probe를 반복한다.
루트에서 선택한 m을 자식 노드 또는 동일 depth의 노드가 우선 재사용하고,
restart stability가 임계값 미만일 때만 local re-scout하는 정책을 실험한다.

이 변경은 K 선택과 노이즈 품질에 영향을 줄 수 있으므로 exact 대비 선택 K,
ARI/NMI, noise 비율, m scout 호출 수를 함께 기록한다.

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

### P2-2. 대형 상태의 envelope 직렬화 경로

checksum envelope는 payload를 bytes로 pickle한 뒤 envelope 자체를 다시 pickle해
atomic write한다. 300건 profile에서는 0.07초로 우선순위가 낮지만, 3,000건 state와
장기 update에서는 큰 bytes 복사·peak memory가 될 수 있다.

payload checksum을 stream 또는 sidecar manifest로 분리하고, checksum·legacy
load·atomic replace 계약을 유지하는지 대형 state에서 측정한다.

## 작업 트래커

| ID | 상태 | 작업 | 선행 조건 | 완료 기준 |
|---|---|---|---|---|
| E-00 | 완료 | 반복 중 `_minimum_center_distance`의 sklearn 거리 호출 제거 | JS 성능 계획 8.2 | 동일 centers/collapse 판정, warm FCM profile에서 범용 거리 호출 제거 |
| N-01 | 대기 | skip-visualization lazy import | 없음 | skip CLI가 UMAP/plotting을 import하지 않고 cluster 결과·state가 기존과 일치 |
| N-02 | 대기 | Python binary cache 및 부분 loader | N-01과 독립 | Gemini 표본 ID/embedding 일치, 전체 JSON materialization 없음, load 시간·RSS baseline 기록 |
| N-03 | 대기 | m scout 재사용 정책 | fast K benchmark | exact 대비 K/ARI/NMI/noise 기준 통과, scout 호출 수와 fit 시간 감소 |
| N-04 | 조사 | restart·sibling Python 병렬화 | thread/BLAS 제어 실험 | worker 1/N의 seed 결과 일치, oversubscription 없음, warm fit p50 개선 |
| N-05 | 조사 | Float32 Python 경로 | 수치 fixture 확장 | labels/중심/XB/update 허용오차 통과, input·state RSS 및 크기 감소 |
| N-06 | 보류 | conditional membership 선택화 | downstream schema 사용처 조사 | opt-in/off 계약 확정, 필요 없는 실행의 rows×paths 배열·열 미생성 |
| N-07 | 보류 | state envelope 복사·직렬화 개선 | 3,000건 이상 state I/O profile | checksum/legacy/atomic 계약 통과, peak RSS·save/load 시간 비교 |

## 실행 순서

1. E-00을 먼저 완료해 warm FCM의 현재 최대 비용을 제거한다.
2. N-01과 N-02를 병렬로 진행해 cold CLI와 표본 실행의 지배적 비용을 줄인다.
3. N-03을 Gemini exact-vs-fast benchmark로 검증한다.
4. N-04와 N-05는 수치 fixture 및 BLAS 환경을 고정한 별도 spike로 판단한다.
5. N-06과 N-07은 출력 계약 및 대형 state profile이 필요할 때만 시작한다.

## 공통 검증 규칙

- 클러스터링·증분 검증 데이터는 `dbpedia_gemini_embeddings.json.gz` 또는 원본
  JSON의 3,000건 Gemini 데이터만 사용한다.
- 빠른 검증은 `--dataset-sample-size`, 고정 seed, `--fast`를 사용한다.
- 알고리즘 변경은 exact 대비 K, ARI/NMI, noise 비율, XB와 중심 통계를 함께
  기록한다.
- 성능 결과에는 commit, 입력 hash, 설정, seed, cold/warm 구분, runtime, peak RSS를
  남긴다.
- 각 항목을 완료하면 이 문서의 상태와 기준선·결과 링크를 갱신한다.
