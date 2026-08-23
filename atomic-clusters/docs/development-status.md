# Atomic Clusters 개발 현황과 아키텍처

기준일: 2026-08-23

이 문서는 `atomic-clusters/` Obsidian plugin의 현재 구현, 수치 파이프라인,
검증 결과와 남은 통합 작업을 기록한다. 숫자는 저장소의 현재 소스와 `/tmp` 실행
보고서를 대조했다. `/tmp` 보고서는 재현용 입력이며 릴리스 산출물은 아니다.

## 목표와 실행 경계

Atomic Clusters의 목표는 Obsidian desktop에서 Markdown vault를 노트 간 의미적
관계에 따라 클러스터링하고, flat leaf cluster를 bottom-up 계층으로 보여 주는
것이다. 노트 본문에서 embedding을 얻는 단계와 수치 클러스터링 단계를 분리한다.

- 수치 파이프라인과 결과 탐색은 네트워크 없이 실행한다. runtime에 package를
  설치하지 않고 Pyodide도 로드하지 않는다.
- 기본 provider는 `Gemini API`지만 사용자가 전송을 확인한 경우에만 노트 본문을
  Google Gemini로 보낸다. API key는 Obsidian `SecretStorage` reference로만 읽고
  plugin settings에 저장하지 않는다. 따라서 Gemini 임베딩 생성 자체는 offline
  작업이 아니다.
- `local` provider와 `multilingual-e5-small` 경계는 있지만 현재 빌드에는 ONNX
  runner와 model asset이 없어 실행할 수 없다. local inference는 후속 작업이다.

## 현재 desktop 구조

```text
Obsidian Vault API
  └─ Markdown 수집·제외 폴더·content hash
       └─ provider/model/content별 embedding cache
            └─ Node worker_thread (cancel/progress 경계)
                 └─ normalize → automatic PCA → umap-js
                      → Rust/WASM HDBSCAN → bottom-up hierarchy
                           └─ cluster-result.json → Cluster Explorer view
```

`main.ts`는 `Build note clusters`, `Open cluster explorer`, `Cancel clustering`
명령과 설정 화면을 제공한다. `NoteStore`가 vault-relative path, title, 내용,
mtime, (지원 runtime에서는 SHA-256, test fallback에서는 stable FNV) content hash를
수집하고, `EmbeddingCache`가
`provider:model:path`와 content hash를 함께 확인하므로 노트가 바뀐 경우에만
재임베딩한다. `NodeClusteringWorker`에는 `INIT/CLUSTER/CANCEL` 요청과
`READY/PROGRESS/RESULT/ERROR` 응답 계약이 있다.

worker가 생성된 WASM asset을 발견하면 `WasmNumericKernel`을 사용한다. asset이
없는 개발/fixture 빌드에서는 같은 orchestration이 deterministic TypeScript
kernel로 동작한다. 이 fallback은 개발 가능성을 위한 것이며 Python/HDBSCAN
수치 parity를 보장하지 않는다.

## 현재 기본값과 파이프라인 의미

### plugin settings 기본값 (`src/main.ts`)

| 항목 | 현재 기본값 | 의미 |
| --- | ---: | --- |
| embedding provider | `gemini` | Gemini 전송 확인 후 768-dimensional embedding |
| Gemini model | `gemini-embedding-2` | SecretStorage reference 기본값은 `gemini-api-key` |
| local model | `multilingual-e5-small` | runner/asset 미포함으로 현재 unavailable |
| excluded folders | `[]` | 모든 Markdown 파일 대상 |
| HDBSCAN `minClusterSize` | `5` | 현재 설정 화면에는 직접 노출하지 않음 |
| HDBSCAN `minSamples` | `3` | core-distance 이웃 수 |
| UMAP `nNeighbors` | `15` | 데이터 수가 작으면 `n - 1`까지 clamp |
| UMAP `minDist` | `0.1` | `umap-js` 설정 |
| `pcaVarianceTarget` | `0.9` | 저장 호환용 legacy 값; 현재 선택 기준은 variance cutoff가 아님 |

### `clusterEmbeddings` 기본값

seed가 지정되지 않으면 `42`를 사용한다. 입력은 non-empty rectangular matrix여야
하고 각 행은 L2 normalize한다.

| 항목 | 현재 기본값 |
| --- | ---: |
| PCA sample | `min(2000, rowCount)` |
| PCA minimum components | `32` |
| PCA maximum components | `512` (embedding dimension과 rank로 제한) |
| PCA knee pilot width | `256` (위 제한 후) |
| PCA candidate step | `32` (`32, 64, ...`) |
| preservation neighbor k | `15`, `30` |
| local minimum gain | `0.05` |
| UMAP components | `20` |
| UMAP neighbors | `15`, 입력 수에 따라 clamp |
| UMAP `minDist` | `0.1` |
| HDBSCAN `minClusterSize` | `5` |
| HDBSCAN `minSamples` | `3` |
| UMAP/HDBSCAN seed | `42` |

정상적인 흐름은 다음과 같다.

1. 입력 embedding을 행별 L2 normalize한다.
2. deterministic sample(기본 2,000행)에 bounded PCA pilot을 한 번 fit한다.
3. pilot projection의 32-component prefix들을 normalize한 뒤, 원래 normalized
   embedding의 exact cosine kNN과 `k=15,30` 평균 보존율을 비교한다.
4. 첫 local gain이 `0.05` 미만이면 이전 prefix를 선택한다. 전체
   monotonized preservation curve의 global knee가 더 뒤이면 그 knee를 선택할
   수 있다.
5. 선택된 폭으로 전체 normalized 입력에 PCA를 한 번 더 fit한다.
6. seeded `umap-js`에서 20차원 표현을 만들고 WASM HDBSCAN이 leaf label,
   probability, outlier score를 계산한다.
7. 각 leaf를 PCA 공간의 probability-weighted normalized center로 바꾸고 가장
   가까운 두 활성 center를 반복 병합해 `leaves`, `merges`, `root`를 만든다.
   noise(`label=-1`)는 hierarchy leaf가 아니다.

### PCA 보정 이력과 현재 one-pilot 의미

초기 실행은 자동 PCA를 variance/후보 계산 방식으로 다루어 실행마다 후보 폭과
결과가 달라졌다. `/tmp/atomic-clusters-offline-3000.json`의 `selected=352`와
`/tmp/atomic-clusters-offline-3000-knee.json`의 `selected=256`은 이전 단계의
결과이며 현재 결과로 사용하지 않는다.

이후 후보별 PCA를 다시 fit하지 않고 하나의 bounded pilot projection을 후보
prefix가 공유하도록 바꾸었다. `/tmp/atomic-pca-knee-3000-single-pilot.json`
(`selected=128`)과 `/tmp/atomic-pca-preservation-3000.json` (`selected=128`)은
이 전환 과정의 측정치다. 마지막으로 preservation 계산의 정규화와 local plateau
선택을 Python 규칙에 맞추면서 `/tmp/atomic-pca-preservation-corrected-3000.json`
은 `selected=96`을 기록했다. 이 보고서들의 cluster 수는 서로 다른 구현 시점의
것이므로 현재 결과 표에 섞지 않는다.

현재 구현(`src/clustering.ts`)의 one-pilot semantics는 다음과 같다.

- 후보 점수는 같은 pilot PCA의 `projected.slice(0, dimension)`을 사용한다.
- 후보 prefix 자체를 다시 L2 normalize하고, 원본 normalized sample을 reference로
  삼아 exact cosine kNN 보존율을 계산한다.
- kernel PCA 호출은 pilot 1회와 최종 전체 입력 1회다. 후보 수만큼 randomized
  PCA를 다시 fit하지 않는다. 이 계약은 `algorithm.test.mjs`가
  `[pilotWidth, selectedWidth]` 호출으로 검증한다.
- 최종 결과의 `pca.candidates`, `preservationCandidates`, `selectionReason`을
  JSON에 남겨 선택 원인을 확인할 수 있다.

## WASM numerical kernels

`atomic-clusters/wasm-core`는 filesystem/network 없는 Rust crate이며 flat
row-major `f32` TypedArray 계약을 사용한다. 현재 export/adapter 경로는 다음을
포함한다.

- `normalize`, `matmul`
- deterministic `pca`와 seeded `randomized_pca`
- `cosine_distances`, tiled exact cosine kNN (`exact_knn_cosine_tiled`)
- deterministic `HnswIndex` (`m=16`, seed `42`)
- sparse mutual-reachability MST와 exact Euclidean HDBSCAN MST
  (`mutual_reachability_mst`, `euclidean_mutual_reachability_mst`)
- condensed-tree HDBSCAN extraction (`hdbscan_extract`, 현재 `selection_method=leaf`,
  `allowSingleCluster=false`)

production adapter의 PCA quality 설정은 `pcaOversamples=16`,
`pcaPowerIterations=3`, `pcaSeed=42`, cosine tile은 `256`이다. HDBSCAN 경로는
Euclidean core distance와 mutual-reachability MST를 WASM 안에서 수행하고
label/probability/outlier score를 JS로 돌려준다. 외부 `hdbscan-rs`
wasm-bindgen provider는 `HdbscanProvider` 교체 지점으로 열어 두었지만 아직
별도 crate를 vendor하고 audit한 상태는 아니다.

## Python authoritative 경로와 JS/WASM 차이

재현 가능한 algorithm reference는 `pca_dimension_search.py`,
`pca_projection.py`, `hdbscan_membership_comparison.py` 및
`pyodide_core/atomic_clustering/`이다. Python 기준은 sklearn
`PCA(svd_solver="full")`로 normalized input을 최대 512폭까지 한 번 fit하고,
32폭 step 후보(가능하면 실제 fitted width도 포함)를 normalized PCA prefix의
cosine kNN 보존율로 평가한다. 기본 k는 15/30, 최소 gain은 0.05, seed는 42다.
Python full-search는 32폭 step 후보를 평가하고, 작은 입력에서는 rank/row 수로
상한을 줄인다. (portable `pyodide_core` selector는 fitted width를 마지막 후보로
추가할 수 있다.) 전체 curve의 global knee를 포함하는 Python full-search와 현재
JS의 256폭 bounded pilot은 계산량/후보 상한이 다르다.

기본 Python discovery는 `umap-learn`의 `UMAP(n_components=20, n_neighbors=15,
init="random", random_state=42, n_jobs=1)`와
`hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean",
cluster_selection_method="leaf", prediction_data=True)`를 사용한다. Python은
native memberships와 probability/outlier score를 보존하고, bottom-up hierarchy
center는 unnormalized selected PCA features와 memberships를 사용한다.

현재 plugin은 다음과 같이 의도적으로 다르다.

- UMAP은 `umap-js@1.4.0`이며 `umap-learn`과 초기화·optimizer·부동소수점 및
  neighbor graph 구현이 동일하다고 가정하지 않는다.
- WASM PCA는 `f32` deterministic randomized kernel일 수 있어 sklearn full PCA와
  basis/설명분산이 bitwise 같지 않다. `randomized_pca`가 없는 asset은 baseline
  `pca` export를 사용한다.
- HDBSCAN은 Python의 native membership matrix를 직접 반환하지 않고 WASM
  `hdbscan_extract`의 label/probability/outlier score를 사용한다. hierarchy도
  JS의 probability-weighted center 규칙으로 별도 생성한다.
- WASM asset이 없으면 TypeScript density-graph fallback이 MST edge percentile
  threshold를 사용한다. 이는 fixture/소규모 개발용이며 Python HDBSCAN parity나
  scientific benchmark로 해석하면 안 된다.

Python 결과는 의미와 configuration의 authoritative alignment 기준이고, offline
report는 실제 plugin orchestration과 WASM 경로의 연결을 확인하는 실행 결과다.
두 구현의 label 번호나 cluster 수가 자동으로 같아진다는 약속은 없다.

## 검증 결과

입력은 모두 `dbpedia_gemini_embeddings.json.gz`의 3,000-record Gemini dataset
(3072 dimensions), sampling seed `42`다. `wasmLoaded=true`인 보고서만 아래에
사용했다. `clustered`는 noise를 제외한 행 수이며 hierarchy merge 수는 leaf 수
보다 하나 작다.

| 보고서 | 행 수/모드 | PCA | leaf clusters | clustered/noise | noise rate | cluster time | hierarchy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `/tmp/atomic-clusters-offline-e2e-final.json` | 100, `--fast` | 64 | 8 | 69 / 31 | 31.0% | 2,422 ms | 8 leaves / 7 merges |
| `/tmp/atomic-clusters-aligned-500.json` | 500, `--fast` | 64 | 45 | 408 / 92 | 18.4% | 14,797 ms | 45 / 44 |
| `/tmp/atomic-clusters-aligned-3000.json` | 3,000, full | 96 | 216 | 2,146 / 854 | 28.47% | 324,111 ms | 216 / 215 |

100/500의 fast 실행은 `pcaSampleSize=min(500,n)`, `pcaMaxComponents=64`,
`umapComponents=10`, `umapNeighbors=10`, `minClusterSize=5`를 사용한다. 따라서
100/500 표는 full default vault 성능이나 Python reference quality와 직접 비교하는
표가 아니다. 3,000 full 실행은 default config(`seed=42`)이며 총 wall time은
load를 포함해 `329,312.6 ms`였다. 500 보고서 총 time은 `20,045.0 ms`, 100 보고서
총 time은 `7,922.3 ms`다. 최신 aligned 결과의 PCA explained-variance fraction은
100: `0.4352`, 500: `0.4733`, 3000: `0.4682`지만 선택 기준은 kNN preservation이다.

이전 `/tmp/atomic-pca-preservation-corrected-3000.json`은 PCA 보정 검증의
중간 산출물(`selected=96`, 111 clusters)이다. 그 결과를 aligned final의 현재
결과로 반복해서 쓰지 않는다. aligned final은 같은 3,000행 입력을 최신 현재
orchestration/asset 조합으로 다시 실행한 `/tmp/atomic-clusters-aligned-3000.json`이다.

## 빌드·테스트 명령

plugin directory(`atomic-clusters/`)에서 실행한다.

```bash
npm install
npm run build
npm test
npm run validate:offline -- --dataset-sample-size 100 --dataset-sample-seed 42 --fast
```

offline runner는 plain `.json` 또는 `.json.gz` Gemini dataset을 읽고,
`--output PATH`가 없으면 `/tmp/atomic-clusters-offline-e2e-*.json`에 JSON report를
만든다. report에는 dataset/options, WASM export 상태, phase progress, timings,
PCA diagnostics, cluster/noise/probability 요약, hierarchy 요약, row-aligned
assignments가 포함된다. validation에는 `dbpedia_label_embeddings.json`을
사용할 수 없다.

WASM asset을 만들고 release bundle을 만들 때는 Rust toolchain이 필요하다.

```bash
npm run build:wasm
npm run build:release
cd wasm-core && cargo test
```

저수준 수동 경로는 `wasm-core/README.md`의 `wasm-bindgen --target web` 절차를
따른다. 릴리스 빌드는 `wasm-core/pkg`가 없으면 실패한다. 문서 변경의 whitespace
검증은 repository root에서 다음을 사용한다.

```bash
git diff --check
```

## 산출물과 저장 위치

- build: `atomic-clusters/dist/main.js`, `dist/manifest.json`, `dist/styles.css`.
  worker source는 `main.js`에 embedded된다.
- generated numerical asset: `atomic-clusters/wasm-core/pkg/`의 wasm-bindgen
  JS/WASM. packaged plugin은 이를 bundle하고 runtime download를 하지 않는다.
- vault runtime cache: `.obsidian/plugins/atomic-clusters/embedding-cache.json`.
  provider/model/path와 content hash별 embedding record를 저장한다.
- vault runtime result: `.obsidian/plugins/atomic-clusters/cluster-result.json`.
  schema version, ids, labels, probabilities, outlier proxy, PCA diagnostics,
  hierarchy, timings를 저장한다.
- Obsidian plugin settings: `.obsidian/plugins/atomic-clusters/settings.json`.
  SecretStorage key 자체는 이 파일에 쓰지 않는다.
- offline validation report: 기본적으로 `/tmp`의 JSON. plugin이 CSV를 생성하는
  경로는 아니며 row-level `assignments`는 report JSON 안에 있다.

## 알려진 제한과 다음 단계

현재의 중요한 제한은 다음과 같다.

- `umap-js`와 authoritative `umap-learn`은 동일 구현이 아니므로 seed가 같아도
  좌표·HDBSCAN 결과가 같다고 볼 수 없다.
- local `multilingual-e5-small`은 UI boundary만 있고 ONNX runtime/model asset이
  없어 실제 offline embedding을 생성하지 못한다.
- Gemini provider는 사용자 확인을 전제로 하는 명시적 network 예외이며,
  embedding 생성과 cache warm-up을 완전 offline으로 만들지는 않는다.
- WASM이 빠진 개발 빌드의 JS density-graph fallback은 HDBSCAN parity 경로가
  아니다. 큰 vault에는 generated WASM을 포함한 release build가 필요하다.
- Rust core의 HDBSCAN extraction은 현재 내부 condensed-tree 구현이다. 외부
  `hdbscan-rs` provider 교체와 Python native membership parity audit은 아직
  끝나지 않았다.
- exact Euclidean MST와 UMAP은 3,000행에서도 측정상 수 분이 걸린다. worker
  cancellation 경계, memory pressure, 더 큰 vault의 UX를 계속 측정해야 한다.

다음 우선순위는 (1) Python/WASM golden fixture로 PCA·UMAP·HDBSCAN 허용오차와
  결과 계약을 고정하고, (2) release build에 WASM asset 검증을 포함하며,
  (3) local ONNX provider와 model download/삭제·disclosure UX를 구현하고,
  (4) 외부 HDBSCAN provider 및 memberships를 audit하고, (5) 3,000행 이상
  vault에서 tiled memory/performance와 progress/cancel UX를 재측정하는 것이다.
