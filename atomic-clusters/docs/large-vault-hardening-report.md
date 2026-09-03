# Large Vault hardening measurement snapshot (release WASM)

실행일: 2026-09-03 UTC
Runner: `scripts/large-vault-hardening.mjs`
출력: `/tmp/atomic-clusters-large-vault-hardening-final-2026-09-03/`

## 입력과 재현성

- 입력: `/workspaces/codespaces-blank/dbpedia_gemini_embeddings.json.gz`
- 원본 행 수: 3,000
- 차원: 3,072
- 압축 입력 SHA-256: `9a949bec1402b52f4b2cba4376ea3eda7c69003b33b7b1ea72e9501cf84d25fc`
- sampling seed: `42`
- clustering seed: `42`
- host: Node `v24.14.0`, Linux `x64`, 2 logical CPUs, Intel Xeon Platinum 8370C

## 결과

| 요청 크기 | 결과 | backend | clustering wall time | peak RSS | 최대 progress gap | main-thread >250ms |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1,000 | measured | `rust-wasm` | 97,745.1 ms | 1,218,125,824 bytes | 88,648.3 ms | 없음 |
| 3,000 | measured | `rust-wasm` | 167,309.7 ms | 1,623,543,808 bytes | 144,482.3 ms | 없음 |
| 5,000 | unavailable | — | — | — | — | 원본 3,000행 |
| 10,000 | unavailable | — | — | — | — | 원본 3,000행 |

runner preflight의 보수적 working-set 추정치는 1,000행 `203,209,472` bytes,
3,000행 `450,441,472` bytes였다. 이는 allocator 보장이 아니라 JS matrix,
worker clone, PCA covariance upper bound, projection working copies를 포함한
추정치다. 실제 run은 입력된 순서대로 각 크기를 한 번씩 실행했다.

## Phase timing과 progress liveness

아래 phase 수치는 `clusterEmbeddings`가 보낸 progress callback 사이의
관찰 가능한 경계 시간이다. PCA 내부 계산 자체의 정확한 시작/종료를 뜻하지
않으며, 이 데이터에서 가장 큰 `pca → umap` 침묵 구간은 30초 기준을 크게
넘었다. 새 renderer heartbeat는 이 구간에도 Notice의 elapsed/“Still working”
상태를 갱신하지만, 측정된 알고리즘 progress gap을 제품 acceptance로
위장하지 않는다.

| 관찰 경계 | 1,000행 | 3,000행 |
| --- | ---: | ---: |
| PCA → UMAP progress gap | 88,648.3 ms | 144,482.3 ms |
| UMAP → HDBSCAN/MST | 4,403.7 ms | 10,296.0 ms |
| HDBSCAN/MST → hierarchy | 217.7 ms | 1,010.1 ms |
| hierarchy → visualization | 1.2 ms | 5.9 ms |
| visualization → complete | 3,875.7 ms | 9,128.1 ms |
| metadata-only title generation | 147.9 ms | 366.0 ms |

worker heartbeat의 최대 간격은 각각 55.4 ms와 55.0 ms였고,
`monitorEventLoopDelay()` 최대치는 43.7 ms와 78.6 ms였다. 두 run 모두
반복적인 250 ms 초과 main-thread stall은 관찰되지 않았다.

## Cancellation boundary probe

동일 Gemini 파일에서 100행을 deterministic sample하고, protocol probe에 한해
임베딩 차원을 64로 잘랐다. 이 수치는 대형 clustering 품질/성능 결과가 아니다.

| Boundary | 취소 latency (ms) |
| --- | ---: |
| PCA | 218.6 |
| UMAP | 137.1 |
| HDBSCAN | 164.6 |
| hierarchy | 162.3 |
| visualization | 156.6 |

각 phase의 첫 progress callback에서 `CANCEL`을 전송하고 다음 cooperative
check에서 `CANCELLED` 오류가 관찰되는지 측정했다. 다섯 경계 모두 취소가
관찰됐다. latency 범위는 137.1–218.6 ms였다.

## WASM 및 검증 상태

`wasm-core/pkg/atomic_clusters_wasm_core.js`와
`atomic_clusters_wasm_core_bg.wasm`가 존재했고, `verifyWasmAsset`이 145,793
bytes와 required exports를 검증했다. 따라서 `releaseWasm` 상태는 `passed`이며
두 primary run의 backend도 `rust-wasm`이었다.

## 검증 명령

- `npm run build`: 통과
- `node --test --test-isolation=none tests/large-vault-hardening.test.mjs`: 7/7 통과
- `npm run benchmark:large-vault -- --output-dir /tmp/atomic-clusters-large-vault-hardening-final-2026-09-03`: 1,000/3,000 measured, release WASM passed
- `npx tsc --noEmit`: 통과
- memory preflight/heartbeat focused tests: 8/8 통과

상세 원본은 `/tmp/atomic-clusters-large-vault-hardening-final-2026-09-03/hardening-report.json`
및 `hardening-report.md`에 있다. release WASM이 생성된 머신에서 같은 명령을
재실행하면 같은 방식으로 1,000행과 3,000행이 순차적으로 측정된다. 5,000행과
10,000행은 실제 입력이 공급될 때까지 unavailable로 남는다.
