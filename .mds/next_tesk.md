# 다음 작업

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

### 1. 드리프트 판정 안정화

현재 새 배치의 natural noise 비율이 임계값을 넘으면 즉시 재클러스터링한다.
작은 배치 하나에 지나치게 민감할 수 있으므로 다음을 추가한다.

- 최소 판정 표본 수
- 최근 배치 natural-noise 비율의 이동 창 또는 EWMA
- 진입/해제 임계값을 분리한 hysteresis
- 재클러스터링 직후 cooldown
- 중심 이동량, 클러스터 점유율 변화, assignment 변경률 진단값
- 급격한 드리프트와 점진적 드리프트에 대한 시계열 테스트

기존 `noise_threshold`, XB degradation 정책과 상태 호환성을 유지해야 한다.
새 설정에는 명시적인 기본값과 상태 마이그레이션을 둔다.

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
cd /tmp/codespaces-blank-incremental-delta
git status --short --branch
python -m unittest discover -v
python -m unittest discover -s tests -v
```

두 번째 테스트 명령의 spherical zero-vector 선행 실패 여부는 `main`과 비교해서
판단한다.
