# Python 알고리즘 최적화 및 JavaScript 이식 계획

## 1. 목표와 전제

먼저 Python 기준 구현에서 알고리즘과 출력 계약을 고정한다. 결과를 바꾸지 않는
시간복잡도 개선을 Python에 적용하고, 필요한 알고리즘 변경은 품질 기준을 통과한 뒤
명시적으로 확정한다. 그 다음 Node.js와 브라우저 Worker에서 공유할 수 있는 TypeScript
코어에 같은 수식, 실행 단계, seed 규칙을 최대한 그대로 이식한다.

NumPy와 scikit-learn의 수치 커널은 이미 네이티브 코드이므로 순수 JavaScript로
옮긴다는 이유만으로 빨라지지는 않는다. Python에서 먼저 다음 개선을 검증하고 확정한
뒤 JS가 동일한 구조를 구현한다.

1. FCM 소속도 계산의 임시 `n × k × k` 배열을 없애고 `O(nk)`로 계산
2. 후보 평가 중 이미 계산한 거리·소속도·라벨을 후속 단계에서 재사용
3. 증분 업데이트에서 영향받은 문서와 노드만 다시 계산
4. 고정된 Python 데이터 계약을 평면 바이너리/TypedArray로 대응
5. 독립 후보 K, 재시작, 형제 노드를 고정 Worker pool에서 병렬 실행

Python 단계에서는 각 최적화 전후의 수치 동등성과 품질을 fixture로 고정한다. JS의 첫
대상은 Node.js CLI이며 수치 코어는 DOM과 Node 전용 API에 의존하지 않게 만든다.
브라우저에서는 동일 코어를 Web Worker로 실행한다. WebGPU와 WASM은 첫 구현의 전제가
아니며, 동일 알고리즘의 TypeScript 구현이 목표를 못 맞춘 커널에만 후속 적용한다.

## 2. 현재 기준선

`main`의 `0fc857b`에서 다음 고정 표본을 `cProfile`로 측정했다.

```bash
python -m cProfile -o /tmp/codespaces_perf_main.prof \
  incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 300 --dataset-sample-seed 42 \
  --state-output /tmp/codespaces_perf_main.state.pkl \
  --pca-components 64 --max-depth 3 \
  --min-node-size 30 --min-child-size 10 --max-clusters 4 \
  --fast --skip-visualization
```

프로파일은 300개만 클러스터링하지만 입력 단계에서는 압축 JSON 3,000개를 모두
파싱한다. `cProfile` 오버헤드와 Python 모듈 import 시간이 포함되므로 아래 값은
절대 성능 목표가 아니라 병목 우선순위를 정하는 자료다.

| 구간 | 누적 시간 | 관찰 |
|---|---:|---|
| 전체 프로세스 | 23.69초 | import 포함 cold run |
| 상태 적합 | 13.01초 | 대부분 계층 FCM |
| 계층 적합 | 12.89초 | 9개 노드의 빠른 K 선택 |
| FCM 후보 선택 | 11.99초 | 51회 FCM, 135회 재시작 |
| 거리 함수 호출 | 6.99초 | 검증/할당을 포함해 11,422회 |
| 입력 로드 | 6.13초 | 전체 gzip JSON materialization |
| JSON decode 자체 | 4.14초 | 표본 추출 전 전체 파싱 |

따라서 Python의 첫 최적화 대상은 FCM 커널과 후보 평가다. 입력 형식은 알고리즘
계약을 동결한 뒤 JS 이식 단계에서 다루며, UMAP이나 렌더링을 먼저 최적화하지 않는다.

### 2.1 진행 순서

최적화와 이식은 다음 네 단계로 분리한다.

1. **Python exact 최적화**: 수학적으로 같은 결과를 내면서 시간·공간복잡도를 낮춘다.
2. **Python 알고리즘 확정**: 표본 silhouette처럼 결과에 영향을 줄 수 있는 변경은
   품질 게이트를 통과한 뒤 기본 알고리즘으로 채택한다.
3. **계약 동결**: 입력, 정규화, seed 파생, 중간 fixture, 출력 schema를 버전화한다.
4. **JS 이식**: 동결된 Python 수식과 단계를 TypedArray/Worker 구조로 옮긴다.

JS 단계에서는 알고리즘을 다시 설계하지 않는다. 성능이 부족하면 같은 계산의 메모리
배치, 병렬 실행, WASM SIMD 여부만 바꾸고 Python과의 품질 계약은 유지한다.

### 2.2 첫 exact 최적화 결과

FCM 소속도 정규화를 pairwise ratio 방식에서 정규화된 역거듭제곱 방식으로 바꿨다.
두 수식은 대수적으로 동일하며 테스트에서 `m=1.4, 2.0, 2.5` 결과가 `1e-12`
허용오차 안에서 일치한다.

```text
기존: n × k × k ratio tensor, O(nk²) 시간/공간
현재: n × k inverse powers, O(nk) 시간/공간
```

`100,000 × 8`, `m=2`, 11회 중앙값 기준으로 61.2ms에서 31.5ms로 약 1.95배
단축됐다. 핵심 임시 배열의 이론상 크기는 Float64 기준 51.2MB에서 6.4MB로
줄었다. 작은 계층 노드에서는 기존 전체 함수와 비슷한 실행 시간을 유지한다.

### 2.3 거리 artifact 재사용 결과

선택된 FCM 결과는 마지막 중심에 대한 제곱 거리 행렬을 선택된 재시작 하나에만
보존한다. 후보 selection의 Xie-Beni 계산, membership/noise 분류, 계층 노드의
거리 threshold와 noise score는 이 artifact를 재사용한다. 따라서 같은 후보의
`X × centers` 거리를 단계마다 다시 계산하지 않는다. 재시작 중 선택되지 않은
결과는 dense artifact를 보존하지 않으며, 외부 또는 구형 결과는 기존 재계산
경로로 fallback한다.

Gemini 3,000건에서 고정 seed `42`, PCA-64, 최대 3계층, `--fast` 설정의
300건 표본 benchmark는 fit 35.2초, 일반 update 4.90초, 선택 refresh 7.76초,
state 16.4MB, peak RSS 0.87GB를 기록했다. refresh에서는 12건만 다시 계산하고
282건을 건너뛰었다. 전체 회귀 테스트 58개가 통과했다.

### 2.4 제곱거리 FCM 반복 결과

구면 FCM의 반복은 입력과 중심이 단위 벡터라는 전제에서
`max(2 - 2 * dot(x, center), 0)` 제곱거리를 직접 사용한다. 따라서 반복마다
`euclidean_distances`와 제곱근을 만들지 않고 제곱거리 membership·목적 함수를
계산한다. 기존 Euclidean 반복과 center, membership, objective가 `1e-10` 이내로
일치하는 회귀 테스트를 추가했다.

재현 명령은 다음과 같다.

```bash
./.venv/bin/python benchmark_spherical_fcm_kernel.py \
  --rows 3000 --dimensions 64 --clusters 4 --repeats 5
```

동일 환경의 5회 중앙값에서 legacy Euclidean loop는 0.575초, 제곱거리 loop는
0.286초로 약 2.01배 빨랐다. UMAP과 I/O까지 포함한 전체 fit 시간은 시스템 부하에
따라 흔들리므로, 이 커널 수치는 전체 pipeline 속도와 분리해 기록한다.

## 3. 측정 규약

먼저 Python 단계별 기준선을 만들고, 계약 동결 뒤 같은 manifest를 JS 하네스가
읽도록 확장한다. 모든 결과에는 commit, runtime, CPU 수, 입력 hash, 설정, seed를
기록한다.

측정 시나리오는 다음과 같다.

| 시나리오 | 데이터 | 측정 항목 |
|---|---|---|
| cold load | 3,000 × 원본 차원 | 압축 해제, decode, TypedArray 생성, peak RSS |
| warm fit | 300/3,000 × PCA-64 | FCM kernel, K 선택, 계층 구성, 직렬화 |
| scale fit | 30,000 × PCA-64 합성 데이터 | 처리량, Worker scaling, peak RSS |
| incremental | 기존 3,000 + 10/100/1,000 | 배정, 중심 갱신, 선택적 refresh |

Node 프로세스 warm-up 3회 후 7회를 측정하고 p50/p95를 보고한다. cold load는 새
프로세스로 7회 측정한다. 전체 시간 하나만 보지 않고 다음 phase timer를 반드시
둔다.

```text
load/decompress/decode
normalize/pca
hierarchy/scout/refine
fcm/distance/membership/center
validity/silhouette
assignment/noise
serialize
```

속도와 함께 `process.memoryUsage()`의 RSS/heap/external 최고값, event-loop 지연,
Worker 수를 기록한다. Worker를 사용한 결과는 동일 입력의 단일 Worker 결과와 같이
남긴다.

## 4. 우선순위 P0: Python 정확성 기준선

### 4.1 수치 동등성 fixture

Python에서 정규화 입력, 초기 소속도 또는 초기 중심, 최종 중심, 소속도, 목적 함수,
라벨을 작은 fixture로 내보낸다. JS 테스트는 같은 초기 상태를 받아 다음 허용오차를
검증한다.

- 중심 최대 절대 오차 `1e-4`
- 소속도 최대 절대 오차 `1e-5`
- 같은 초기 상태에서 hard label 일치
- exact 모드의 선택 K와 트리 stop reason 일치
- noise 비율 차이 1%p 이하

PRNG 차이 때문에 seed만 같게 두고 결과 벡터의 완전 일치를 요구하지 않는다. 알고리즘
변경이 들어간 fast 모드는 Python 결과 대비 ARI `0.98` 이상 또는 NMI 차이 `0.02`
이하를 통과 기준으로 삼는다.

### 4.2 JS 이식용 메모리 계약

핫 경로에서는 `number[][]`, 문서별 객체, 문자열 cluster path를 사용하지 않는다.

```ts
type MatrixF32 = {
  data: Float32Array; // row-major
  rows: number;
  cols: number;
};

type NodeSlice = {
  rowIndices: Int32Array;
  depth: number;
  parent: number;
};
```

- 임베딩, PCA 결과, 소속도는 `Float32Array`
- 중심의 가중 합, norm, 목적 함수는 `Float64Array` 또는 number 누산
- 라벨과 행 인덱스는 `Int32Array`
- boolean 상태는 `Uint8Array`
- 트리 문자열 경로는 결과를 내보낼 때 한 번만 생성
- 반복마다 새 배열을 만들지 않고 Worker별 scratch buffer 재사용

정확도 fixture를 통과하지 못하는 구간만 Float64 저장으로 올린다.

## 5. JS 단계 P1: 입력과 상태 형식

현재 gzip JSON은 부분 표본을 쓰더라도 전체 3,000개와 모든 숫자를 객체로 만든다.
호환 loader는 유지하되 hot path는 한 번 변환한 바이너리를 사용한다.

권장 형식은 다음과 같다.

```text
dataset.manifest.json  # shape, dtype, byte order, metadata/state version
dataset.embeddings.f32 # little-endian, row-major Float32
dataset.metadata.ndjson 또는 Arrow IPC
```

Node에서는 `Buffer`가 소유한 `ArrayBuffer` 위에 `Float32Array` view를 만들고 불필요한
복사를 하지 않는다. Worker에는 원본 행렬을 `SharedArrayBuffer`로 한 번만 공유한다.
브라우저에서는 파일 stream으로 읽어 같은 레이아웃을 만든다.

상태 파일도 memberships, centers, PCA 행렬, 좌표를 JSON 배열로 저장하지 않는다.
작은 manifest와 typed binary blob을 분리하고 checksum과 명시적 version을 둔다.

완료 기준:

- 바이너리 cold load p50이 gzip JSON 호환 loader의 35% 이하
- load peak RSS가 gzip JSON 경로의 50% 이하
- 3,000개 중 300개 표본 실행에서 선택 행만 materialize
- Python/JS 양방향 fixture converter 제공

## 6. Python 단계 P0: FCM 커널 복잡도

### 6.1 구면 거리

입력과 중심은 단위 벡터이므로 제곱 거리만 계산한다.

```text
d2(i, j) = max(2 - 2 * dot(x_i, c_j), 0)
```

제곱근, 일반 유클리드 거리 라이브러리, 호출마다 shape/finite 검증을 반복하지 않는다.
검증은 public API 경계에서 한 번 하고 내부 커널은 검증된 평면 buffer만 받는다.

### 6.2 `O(nk)` 소속도 계산

현재 수식 그대로 비율 배열을 만들면 샘플마다 `k × k` 임시 공간이 필요하다. 제곱
거리의 지수를 `q = 1 / (m - 1)`로 두면 같은 값을 다음처럼 계산할 수 있다.

```text
inv(i, j) = d2(i, j)^(-q)
u(i, j) = inv(i, j) / sum_l(inv(i, l))
```

정확히 일치하는 중심이 있으면 해당 중심들에만 `1 / tieCount`를 준다. 이 방식은 기존
비율식과 대수적으로 동일하며 시간과 임시 공간을 `O(nk²)`에서 `O(nk)`로 줄인다.

### 6.3 융합 루프와 buffer 재사용

한 row block을 순회하며 다음을 한 커널에서 처리한다.

1. 모든 중심과 dot product 및 제곱 거리 계산
2. 소속도 정규화
3. 이전 소속도와 최대 변화량 계산
4. hard label과 선택 중심 거리 계산
5. 필요할 때만 목적 함수 누적

두 개의 membership buffer를 번갈아 쓰고 매 반복 `copy`, `map`, spread, 중첩 배열을
만들지 않는다. `m=2`는 `u*u`와 reciprocal만 쓰는 전용 경로를 둔다. 중심 업데이트는
row-major 순회 중 `Float64Array(k * d)`에 누적하고 마지막에 정규화한다.

선정된 후보가 이미 가진 distances, memberships, labels, cluster counts를 noise 판정,
거리 임계값, assignment 생성에 전달한다. 선정 직후 같은 거리 행렬을 다시 계산하지
않는다. 큰 노드에서는 전체 거리 행렬을 보존하기보다 선택 중심 거리와 필요한 요약만
남겨 메모리를 제한한다.

완료 기준:

- fixture 정확도 통과
- `3000 × 64`, K 2~4에서 literal TypeScript 기준선보다 FCM kernel p50 2배 이상 향상
- 반복 중 할당량이 입력 크기에 비례해 계속 증가하지 않음
- 최종 목적 함수와 수렴 iteration 수 기록

## 7. JS 단계 P1: 고정 Worker pool

병렬화 단위는 안쪽 dot-product가 아니라 독립 작업이다.

- 서로 다른 후보 K
- 같은 K의 독립 재시작
- 부모 분할이 끝난 뒤의 형제 노드
- PCA/UMAP과 무관한 평가 작업

프로세스 시작 시 고정 Worker pool을 만들고 실행마다 Worker를 새로 만들지 않는다.
원본/PCA 행렬은 `SharedArrayBuffer`로 읽기 전용 공유하고, 작업 메시지에는 설정, seed,
행 인덱스 view 정보만 보낸다. 결과는 중심, 요약 지표, 필요 시 선택 후보의 소속도만
전송한다.

재현성을 위해 작업 순서와 무관하게 seed를 다음 값으로 결정한다.

```text
seed = hash(globalSeed, nodeId, candidateK, restartIndex, phase)
```

작은 노드는 직렬 실행한다. 기준선 측정으로 `rows × dims × k` 임계값을 정하고 Worker
왕복 비용보다 계산량이 클 때만 queue에 넣는다. 물리 코어 수보다 많은 CPU 작업을
동시에 실행하지 않는다.

완료 기준:

- 4코어 이상에서 3,000개 warm fit이 단일 Worker 대비 1.5배 이상 향상
- Worker 수 증가에 따라 peak RSS가 원본 행렬 크기만큼 반복 증가하지 않음
- Worker 수 1과 N에서 동일 seed의 선택 결과가 같음

## 8. Python 단계 P1: 탐색량과 중복 계산 줄이기

### 8.1 두 단계 K 선택

모든 후보와 재시작에 exact silhouette를 계산하지 않는다.

1. 고정 seed의 최대 1,000개 표본으로 K와 m scout
2. scout에서는 `O(nk)` 중심거리 silhouette proxy를 사용하고, full-data refine에서만 exact silhouette를 계산
3. scout 점수 차가 작을 때만 상위 2개 K를 전체 데이터로 refine하고, 명확한 1위는 해당 K만 refine
4. 점수 차이가 허용오차 안일 때만 추가 K 또는 재시작 수행
5. 최종 후보에만 exact metric과 noise threshold 계산

exact 모드는 회귀 검증용으로 계속 제공한다. fast 모드는 각 생략 이유와 표본 크기,
후보 점수 차이를 결과에 기록해 품질 저하를 추적한다.

#### `refine_score_margin` 검증 결과

2026-08-08에 `benchmark_fast_fcm_selection.py`로 Gemini 임베딩 3,000건
(원본 3,072차원, gzip)을 고정 seed `42`로 표본화하고, PCA-64·K 2~4·최소
자식 크기 20·exact `n_init=10` 조건에서 exact selector와 fast selector의
선택 K를 비교했다. selector seed는 `42, 43, 44`이며 margin `0.15`와 기존
상위 2개 refine에 해당하는 `1.0`을 함께 측정했다.

| 표본 수 | margin | K 일치율 | 평균 refine K 수 | fast/exact 시간비 | 평균 label ARI |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.15 | 66.7% | 1.33 | 1.50x | 0.635 |
| 100 | 1.0 | 33.3% | 2.00 | 1.41x | 0.538 |
| 300 | 0.15 | 100% | 1.67 | 1.20x | 1.000 |
| 300 | 1.0 | 100% | 2.00 | 1.50x | 1.000 |
| 1,000 | 0.15 | 100% | 1.00 | 1.16x | 1.000 |
| 1,000 | 1.0 | 100% | 2.00 | 1.25x | 1.000 |
| 3,000 | 0.15 | 100% | 1.00 | 1.27x | 1.000 |
| 3,000 | 1.0 | 100% | 2.00 | 2.06x | 1.000 |

100건 표본만 seed를 `42~51`로 늘린 추가 측정에서는 margin `0.15`의 K
일치율이 60%, margin `1.0`은 50%였다. 작은 표본에서는 scout 자체의 K
선택 변동이 남아 있으므로 이 결과를 전체 fast 모드의 정확성 보장으로 해석하지
않는다. 다만 300건 이상에서는 3개 seed 모두 exact K와 일치했고, 3,000건에서
refine 수를 평균 2개에서 1개로 줄이면서 label ARI를 유지했다. 따라서
`0.15`를 fast path의 기본값으로 채택하되, 작은 노드에서 보수적인 탐색이 필요하면
`FastFcmConfig(refine_score_margin=1.0)`으로 기존 상위 2개 refine 동작을 선택할
수 있게 한다.

재현 명령:

```bash
./.venv/bin/python benchmark_fast_fcm_selection.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --output-json /tmp/fast-fcm-selection-gemini.json \
  --output-csv /tmp/fast-fcm-selection-gemini.csv \
  --dataset-sample-sizes 100 300 1000 3000 \
  --dataset-sample-seed 42 \
  --seeds 42 43 44 \
  --pca-components 64 \
  --max-clusters 4 \
  --min-child-size 20 \
  --exact-n-init 10 \
  --exact-max-attempts 30 \
  --refine-score-margins 0.15 1.0
```

### 8.2 재시작 조기 종료

- 최소 유효 재시작 수를 채운 뒤 안정성이 목표 이상이면 종료
- 반복 5회 이후 중심 붕괴를 검사하되 매 iteration의 범용 거리 API 호출 제거
- 목적 함수 개선과 membership 변화가 함께 정체되면 종료
- 실패한 초기화가 연속될 때 작은 K fallback을 명시적으로 기록

### 8.3 PCA와 k-NN

제품 기본 경로는 검증된 고정 PCA-64 또는 저장된 projection을 사용한다. 자동 차원
탐색이 필요할 때는 최대 폭 PCA를 한 번만 계산하고 prefix를 공유한다. 각 prefix마다
전체 brute-force k-NN을 반복하지 않고 고정 표본 또는 HNSW 기반 근사 이웃을 사용한다.
PCA 자체는 검증된 randomized SVD/WASM 구현을 우선하고, 순수 JS full SVD를 새로
작성하지 않는다.

WebGPU는 `n × d`가 충분히 큰 PCA/dot-product batch에서만 별도 실험한다. 작은 증분
배치나 작은 계층 노드에는 upload와 shader 준비 비용 때문에 사용하지 않는다.

완료 기준:

- fast 모드 품질 기준(ARI/NMI/noise 비율) 통과
- exact 모드 대비 3,000개 warm fit p50 2배 이상 향상
- 선택 K가 달라진 실행에는 scout 근거와 exact 재검증 결과가 남음

## 9. Python 단계 P1: 증분 업데이트

전체 문서의 소속도를 고정 주기마다 다시 계산하는 대신 중심 이동량으로 영향 범위를
결정한다.

- 노드별 이전 중심과 현재 중심의 이동량 저장
- 이동량이 작은 노드는 refresh 생략
- 문서의 기존 fuzzy weight × 중심 이동량이 임계값을 넘을 때만 재배정
- append-only batch는 tree count를 delta로 갱신
- replacement batch만 ID index를 이용해 기존 기여도를 빼고 새 기여도를 더함
- drift/noise/XB는 작은 배치를 누적해 판정하고 반복 full recluster에 cooldown 적용

문서 ID lookup은 `Map<Id, rowIndex>` 하나로 유지한다. 문서별 중첩 객체 대신 node,
cluster, weight를 평면 sparse buffer로 저장한다.

완료 기준:

- 기존 3,000개에 10개 추가 시 전체 refresh 경로 시간의 20% 이하
- selective 결과와 full refresh 결과의 assignment 일치율 99% 이상
- 중심, count, 충분통계량이 full recompute fixture와 허용오차 내 일치
- 메모리가 업데이트 횟수에 따라 누수 없이 `O(totalRows + nodes × k × d)` 유지

## 10. 구현 순서와 게이트

| 단계 | 산출물 | 다음 단계 진입 조건 |
|---|---|---|
| 1 | Python benchmark와 수치 fixture | phase별 기준선과 품질 지표 재현 |
| 2 | Python `O(nk)` membership과 제곱거리 FCM 반복 (완료) | 수치 동등성 및 메모리 목표 통과 |
| 3 | Python 거리/metric artifact 재사용 (완료) | exact 결과 유지, 전체 fit 회귀 없음 |
| 4 | Python fast K·silhouette 알고리즘 확정 | 품질 및 exact 대비 목표 통과 |
| 5 | Python 계층/증분 선택적 재계산 | assignment/통계 fixture 통과 |
| 6 | 알고리즘·입출력 계약 v1 동결 | Python fixture와 schema 버전 확정 |
| 7 | JS binary loader와 TypedArray 단일 Worker | Python fixture 정확도 통과 |
| 8 | JS K/restart Worker pool | 결정성 및 1.5배 scaling 통과 |
| 9 | PCA/UMAP 및 브라우저 통합 | Node 기준선 회귀 없음 |
| 10 | 필요 시 WASM SIMD/WebGPU spike | JS가 전체 목표를 못 맞춘 커널만 채택 |

각 단계는 직전 단계 benchmark JSON을 보존하고 같은 입력으로 비교한다. 목표를 못 맞춘
최적화는 복잡도를 더하기 전에 allocation profile, Worker overhead, 캐시 miss 여부를
확인한다.

## 11. 최종 성능 목표

최초의 정확한 literal TypeScript 포트를 JS 기준선으로 삼는다.

- optimized JS warm fit p50: literal JS의 50% 이하
- 3,000개 fixed PCA-64 fast fit: Python 기준선의 1.2배 이내
- binary cold load p50: gzip JSON 호환 경로의 35% 이하
- peak RSS: literal JS의 60% 이하
- 10개 증분 batch: 전체 refresh의 20% 이하
- 결과 품질: exact fixture 통과, fast ARI `>= 0.98` 또는 NMI 차이 `<= 0.02`

TypedArray 단일 Worker 구현이 FCM 목표를 못 맞추면 그때 FCM dot/center kernel만 WASM
SIMD 후보로 올린다. WebGPU는 30,000개 scale benchmark에서 CPU Worker pool을 유의미하게
이길 때만 채택한다.

## 12. 의도적으로 피할 것

- `Array<Array<number>>` 기반 행렬과 hot loop의 `map/reduce`
- 후보마다 입력 행렬을 Worker로 복사
- 요청마다 Worker 생성
- 정확한 silhouette를 모든 scout 후보에 반복 계산
- 전체 JSON 상태에 embeddings/memberships를 숫자 배열로 저장
- 측정 전에 WebGPU, WASM, GPU 라이브러리부터 도입
- Python과 JS의 cold import/startup 시간을 warm kernel 시간과 섞어 비교
