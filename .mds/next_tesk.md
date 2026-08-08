# 다음 작업

## 2026-08-06 중심 영향 기반 선택 membership 갱신 완료

- 마지막 membership 갱신 당시의 계층 중심을 상태에 스냅샷으로 저장한다.
- 현재 중심과 스냅샷의 누적 이동량이 기본 `0.01` 이상인 클러스터만 영향
  후보로 삼는다.
- `중심 이동량 × 저장된 membership ** m`이 기본 `0.05` 이상인 문서와
  신규·교체 문서만 membership과 compact contribution을 다시 계산한다.
- 영향받지 않은 문서의 assignment와 contribution 객체는 그대로 공유한다.
- 이동 노드의 거리 임계값과 계층 XB는 저장된 PCA 투영값·fuzzy weight로
  계산해 전 문서 membership 호출을 피한다.
- 실제 noise/XB 재클러스터링이 발동한 경우에만 전체 데이터를 다시 계산한다.
- 상태 버전 6에 중심 스냅샷을 추가했고, v1~v5 상태는 현재 중심으로 스냅샷을
  초기화한다. 구형 상태는 기존 전 문서 refresh를 유지한다.

검증 항목:

- 이동 중심에 fuzzy weight가 큰 문서만 선택되는지 확인
- 선택 refresh에서 신규 2개만 다시 계산하고 기존 6개는 건너뛰는지 확인
- 전 문서 거리 임계값 refresh가 호출되지 않는지 확인
- fresh contribution 기반 XB와 정확한 전체 XB가 `1e-12`까지 일치하는지 확인
- 구형 상태의 `full_legacy` fallback 확인
- 문서 임베딩 3,000개 중 100개를 `--fast` 적합한 뒤 신규 10개를 갱신한
  실제 스모크에서 전체 110개 중 33개만 membership 갱신하고 77개를 건너뜀
- 루트 테스트 53개 통과, `tests/`는 20개 중 기존 zero-length 실패를 제외한
  19개 통과

## 2026-08-06 드리프트 판정 안정화 완료

우선순위 1의 드리프트 판정 안정화를 구현했다.

- 작은 배치는 natural-noise 개수와 표본 수를 누적하고, 기본 20개부터 판정한다.
- 판정 비율에 기본 `alpha=0.30` EWMA를 적용한다.
- 기본 진입 5%, 해제 2.5%의 hysteresis를 적용한다.
- 재클러스터링 직후 기본 3회 업데이트 cooldown을 적용하며 noise와 XB 트리거를
  함께 억제한다.
- 중심 이동 평균/최대, 클러스터 점유율 총변동거리, 기존 ID assignment 변경률을
  업데이트 요약과 상태에 저장한다.
- 상태 버전을 6으로 올렸다. v1~v5 상태는 기존 즉시 판정 동작을 유지하도록
  명시적으로 마이그레이션한다.
- 급격한 소규모 배치 누적, 점진적 EWMA 상승, hysteresis 해제, 반복 트리거
  cooldown 테스트를 추가했다.

검증 결과:

- 루트 테스트 49개 통과
- `tests/` 테스트 20개 중 19개 통과
- 남은 zero-length spherical FCM 테스트는 이 브랜치가 갈라진 뒤 `main`의
  `1075ae7`에서 수정된 항목이므로 브랜치 동기화 시 반영한다.
- `dbpedia_gemini_embeddings.json` 문서 3,000개 중 100개를 고정 시드로 추출한
  `--fast --skip-visualization` 적합 통과

빠른 검증은 전체 데이터 대신 고정 시드 축소 표본과 `--fast`를 사용한다.

```bash
./.venv/bin/python incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json \
  --dataset-sample-size 100 \
  --dataset-sample-seed 42 \
  --fast \
  --pca-components 32 \
  --max-depth 2 \
  --min-node-size 30 \
  --min-child-size 10 \
  --max-clusters 4 \
  --state-output /tmp/incremental-drift-fast-smoke.pkl \
  --skip-visualization
```

## 2026-08-05 완료 상태

작업 브랜치는 `refactor/incremental-delta-updates`, 워크트리는
`/tmp/codespaces-blank-incremental-delta`다.

증분 클러스터링의 1차 고도화를 완료했다.

- 문서마다 계층 노드별 `clusters x PCA dimensions` outer product를 저장하던
  방식을 제거했다.
- 문서별 PCA 투영 벡터 하나와 경로별 fuzzy weight만 저장하는
  `compact_weights_v1` 형식을 도입했다.
- 일반 업데이트는 전체 contribution을 `deepcopy`하고 재합산하지 않는다.
  변경 ID의 이전 contribution을 aggregate에서 빼고 새 contribution을 더한다.
- 손대지 않은 문서 contribution은 새 상태와 공유하며, 작은 노드별 aggregate만
  복사한다.
- membership 전체 갱신과 재클러스터링 시에만 전체 contribution을 다시 계산한다.
- 저장 상태 버전을 4로 올렸다. v1~v3 상태와 과거 outer-product contribution은
  계속 읽을 수 있고, 첫 업데이트 또는 저장 시 compact 형식으로 변환한다.
- 기본 파이프라인에서 업데이트 메타데이터에 새 열이 추가될 때 발생하던 pandas
  dtype 충돌을 수정했다.

검증한 항목:

- 신규 ID 추가와 기존 ID 교체
- 일반 배치에서 전체 contribution 재합산이 호출되지 않는지 확인
- 20개 연속 배치 후 delta aggregate와 전체 재계산 결과가 `1e-12` 범위에서
  일치하는지 확인
- 구형 outer-product contribution 자동 변환
- 상태 v4 저장 및 재로딩
- 업데이트 중 메타데이터 스키마 확장
- 전체 고유 테스트 65개 통과

당시 `tests/test_spherical_fcm.py`의 zero-length sample 검증은 브랜치 분기 후
`main`의 수정과 비교해야 하는 항목으로 기록했으며, 현재 브랜치의 전체 테스트에서는
통과한다.

## 이전 우선순위 기록

### 1. 드리프트 판정 안정화 (완료)

다음 항목을 완료했다.

- 최소 판정 표본 수
- 최근 배치 natural-noise 비율의 이동 창 또는 EWMA
- 진입/해제 임계값을 분리한 hysteresis
- 재클러스터링 직후 cooldown
- 중심 이동량, 클러스터 점유율 변화, assignment 변경률 진단값
- 급격한 드리프트와 점진적 드리프트에 대한 시계열 테스트

기존 `noise_threshold`, XB degradation 정책과 상태 호환성을 유지했고 새 설정의
기본값과 상태 마이그레이션을 추가했다.

## 2026-08-07 공통 증분 코어와 운영 안정성 완료

flat `full_pipeline.update_auto_pca_sfcm`과 계층형
`incremental_clustering.update_incremental_state`가
`incremental_core.py`의 공통 batch 처리 기능을 사용한다.

- ID 기반 append/replace와 메타데이터 스키마 확장
- 명시적 또는 content-derived batch ID
- 동일 batch 재실행 멱등 처리와 다른 내용의 ID 재사용 거부
- 상태 generation과 최대 256개 batch replay 기록
- SHA-256 checksum envelope와 legacy raw pickle 읽기 호환
- process-specific 임시 파일을 사용하는 atomic save
- CLI update의 sibling lock을 통한 동시 update 직렬화
- 상태 metadata/assignment ID, 좌표, generation, replay history 검증
- update 실패 시 atomic save 이전 상태 보존

추가한 성능 측정 도구:

```bash
./.venv/bin/python benchmark_incremental_updates.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --update-size 10 \
  --fast \
  --output-json /tmp/incremental-benchmark.json
```

이 도구는 fit/update/refresh 시간, 선택 refresh 표본 수, 상태 파일 크기,
peak RSS를 JSON으로 기록한다.

## 2026-08-08 Gemini benchmark 및 main 통합 진행

`dbpedia_gemini_embeddings.json` 3,000건 전체를 사용해 3072차원 입력의
규모별 baseline을 측정했다. 모든 실행은 seed `42`, `--fast`, 시각화를 포함한
동일 설정이며 update 크기는 전체의 4%로 맞췄다.

| 입력 | fit | 일반 update | 선택 refresh | refresh/skip | state | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 26.5초 | 4.07초 | 2.32초 | 15 / 80 | 5.5MB | 0.87GB |
| 500 | 31.2초 | 4.34초 | 0.47초 | 86 / 404 | 25.2MB | 0.87GB |
| 1,000 | 39.2초 | 4.94초 | 0.37초 | 40 / 940 | 38.5MB | 0.97GB |
| 3,000 | 77.7초 | 5.43초 | 1.51초 | 120 / 2,820 | 88.4MB | 1.36GB |

`main`의 문서 이동·gzip 데이터·spherical FCM 검증 변경은 병합 커밋
`7f74533`으로 현재 브랜치에 통합했고, `.gitignore` 충돌은 양쪽 설정을
보존하는 방식으로 해결했다. 통합 후 루트 테스트 57개와 gzip 100건 fit smoke가
통과했다.

로컬 `main` fast-forward와 원격 `refactor/incremental-delta-updates`, `main`
push까지 완료했다. 이 증분 리팩터 범위의 작업 로그상 필수 작업은 종료되었다.

## 재개 시 확인 명령

```bash
cd /workspaces/codespaces-blank
git status --short --branch
./.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

두 번째 테스트 명령의 spherical zero-vector 선행 실패 여부는 `main`과 비교해서
판단한다.
