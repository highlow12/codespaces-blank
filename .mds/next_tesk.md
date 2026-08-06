# 다음 작업

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

`tests/test_spherical_fcm.py`의 zero-length sample 테스트 1개는 현재 `main`에서도
동일하게 실패하는 기존 문제이며 이번 작업의 회귀가 아니다.

## 다음 작업 우선순위

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

### 2. 두 증분 엔진 통합

`full_pipeline.update_auto_pca_sfcm`은 flat 전체 데이터 갱신이고,
`incremental_clustering.update_incremental_state`는 계층형 delta/XB/noise 정책이다.
공통 증분 코어를 추출해 다음 동작을 한곳에서 관리한다.

- ID 기반 append/replace
- compact contribution delta
- membership refresh 스케줄
- drift 및 재클러스터링 결정
- 상태 버전과 요약 지표

flat 파이프라인과 계층 파이프라인은 모델 adapter만 다르게 두는 방향이 적합하다.

### 3. 운영 안정성

- batch ID를 저장해 동일 배치 재실행을 멱등 처리
- 상태 파일 checksum과 더 엄격한 스키마 검증
- 동시 update 방지를 위한 파일 잠금 또는 generation compare-and-swap
- 실패한 재클러스터링에서 이전 상태로 돌아가는 rollback 테스트

### 4. 성능 검증

문서 수, PCA 차원, 계층 깊이, K를 변화시키며 다음을 측정한다.

- 저장 상태 크기
- 일반 배치 update 시간
- membership refresh 시간
- peak memory

compact 형식과 과거 outer-product 형식의 메모리 비교 결과도 기록한다.

## 재개 시 확인 명령

```bash
cd /workspaces/codespaces-blank
git status --short --branch
./.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

두 번째 테스트 명령의 spherical zero-vector 선행 실패 여부는 `main`과 비교해서
판단한다.
