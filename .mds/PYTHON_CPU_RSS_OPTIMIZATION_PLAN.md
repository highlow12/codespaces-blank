# Python CPU·RSS 공동 최적화 실행 계획

## 목적과 원칙

이 계획은 `incremental_clustering.py`의 실행 시간을 줄이면서 peak RSS도 함께
낮추는 것을 목표로 한다. 한쪽 지표의 개선을 위해 다른 지표가 의미 있게 악화되는
변경은 채택하지 않는다. 알고리즘 결과의 동등성, 재현성, 기존 상태 파일 호환성은
성능보다 우선한다.

현재 우선순위는 관측된 병목과 이미 완료된 작업을 반영한다.

- 입력 JSON 캐시(N-02)는 이미 cold load와 RSS를 크게 줄였다.
- `float32` 혼합 저장(N-05)은 상태 크기와 RSS를 크게 낮췄다.
- 조건부 level membership 출력(N-06)은 기본 경로의 상태/출력 부하를 낮췄지만,
  전체 실행 시간 개선은 측정 잡음 범위였다.
- 2 CPU 환경의 Python worker 병렬화(N-04)는 오히려 느렸으므로 보류한다.

따라서 다음 작업은 큰 배열을 반복 생성하는 연산 경로, PCA 탐색의 중복 계산,
증분 상태의 객체 오버헤드 순으로 다룬다.

## 공통 측정 기준

### 대상과 실행 조건

- 검증 데이터는 3,000-record Gemini embedding 데이터의 JSON 캐시를 사용한다.
  `dbpedia_label_embeddings.json`은 사용하지 않는다.
- warm-run 측정에서는 같은 입력 캐시, 커밋, Python/NumPy/BLAS 환경, seed,
  스레드 설정을 유지한다.
- 기본 표본은 1,000 rows이며, 채택 전에는 3,000 rows로 재측정한다.
- 각 구성은 새 프로세스에서 3회 실행해 중앙값(p50)을 기록한다. 첫 실행만의
  import/cache 효과는 warm CPU 비교에 포함하지 않는다.
- peak RSS는 각 실행 프로세스에서 기록한다. `cProfile`은 시간 분해용 별도 실행으로
  사용하며, 프로파일러 실행 시간을 기준 성능으로 쓰지 않는다.

대표 명령 형식은 다음과 같다. 실제 cache/state 경로와 seed는 결과 표에 함께 남긴다.

```bash
/workspaces/codespaces-blank/.venv/bin/python incremental_clustering.py fit \
  --input-cache <gemini-cache> \
  --dataset-sample-size <1000-or-3000> \
  --dataset-sample-seed 42 \
  --fast --skip-visualization \
  --state-output <state-path>
```

### 결과 동등성 및 채택 기준

성능 수치는 동일 표본·동일 seed에서 기준 커밋과 후보 커밋을 비교한다.

| 항목 | 필수 확인 |
| --- | --- |
| CPU 시간 | 1,000과 3,000 rows의 p50 wall time, 병목 함수 누적 시간 |
| 메모리 | peak RSS와 저장된 state 파일 크기 |
| 클러스터 품질 | 선택된 K, noise 수, XB/품질 점수, 유효 label 분포 |
| 수치 동등성 | 동일 입력에서 중심값/점수/할당이 허용 오차 내 일치하는지 |
| 상태 호환성 | save/load, legacy 상태 migration, incremental replace/idempotency |
| 회귀 방지 | 전체 `unittest` 및 관련 단위·통합 테스트 |

원칙적 채택선은 warm CPU 시간 또는 peak RSS 중 하나에서 **10% 이상** 개선하고,
다른 지표를 **3% 초과** 악화시키지 않는 것이다. 작은 개선은 측정 오차와 코드
복잡도를 고려해 보류한다. 결과가 의도적으로 달라지는 탐색 최적화는 위 표의 품질
지표가 기준과 동등함을 별도로 입증해야 한다.

## 우선순위와 상세 작업

### R-00 — warm CPU·RSS 기준선 및 병목 프로파일

**목표:** 이후 변경의 비교 기준을 고정하고, 실제 CPU·할당 상위 병목을 확인한다.

**범위:** Gemini cache 1,000/3,000 rows의 `fit --fast --skip-visualization` 경로다.
입력 cache 생성 시간, 시각화, 네트워크, JSON 원본 cold read는 이 기준선의 범위에서
제외한다.

**수행 절차:**

1. 현재 커밋, cache manifest/hash, Python·NumPy·BLAS 버전, CPU 수, 관련 환경 변수,
   명령행 인자를 결과 문서에 기록한다.
2. 1,000과 3,000 rows 각각을 새 프로세스 3회 실행하여 p50 wall time, peak RSS,
   state 크기를 기록한다.
3. 같은 입력에서 `cProfile`과 가능한 할당/메모리 프로파일을 별도 수행한다.
4. FCM 후보 K·restart, PCA 탐색/투영, center contribution, state 직렬화의 누적 시간과
   큰 배열/객체의 생성 위치를 구분한다.

**완료 조건:** 재현 가능한 baseline 표와 CPU 상위 함수, RSS 상위 보유 구조가 남고,
C-01/C-02/M-01의 구현 순서를 프로파일 근거로 확인한다.

**중단 기준:** 같은 환경에서 3회 편차가 5%를 넘으면 CPU governor, BLAS thread,
동시 실행 여부를 먼저 통제하고 기준선을 다시 만든다.

### C-01 — FCM candidate K·restart 배열/계산 재사용

**목표:** FCM의 후보 K 및 restart 반복에서 `n_samples × k` 중간 배열을 불필요하게
재생성하거나 동일 값을 다시 계산하는 비용을 줄인다. CPU 최우선 작업이며 RSS는
workspace 수명 관리로 함께 제한한다.

**조사 범위:** `fcm_core.py`의 `spherical_fcm`, 후보 K 선택, restart 점수화 경로와
membership/distance/label 계산을 살핀다. R-00에서 실제 상위 항목인지 먼저 확인한다.

**설계 단계:**

1. 각 후보 K와 restart에서 생성되는 distance, membership, squared distance,
   labels, objective/validity 입력의 shape·dtype·수명을 표로 작성한다.
2. 같은 candidate 안에서 여러 평가 함수가 동일 distance/membership을 다시 만드는
   지점을 찾는다.
3. 결과에 필요한 artifact만 반환·재사용하도록 경계를 정한다. 선택되지 않은
   candidate의 대형 배열은 즉시 해제 가능해야 한다.
4. 프로파일에서 할당이 큰 경우에만 최대 K 기준 workspace의 선할당/재초기화를
   도입한다. 각 K의 유효 열만 사용하고 stale 열이 점수에 섞이지 않게 한다.
5. cross-K 또는 cross-restart의 난수·수렴 결과를 공유하지 않는다. 이는 알고리즘
   의미를 바꿀 수 있으므로 금지한다.

**검증:** 고정 seed로 K 선택, FCM 목적함수, center, label/membership 및 validity
score를 기준 구현과 비교한다. 1,000/3,000 rows에서 CPU p50과 peak RSS를 함께
측정하고, 큰 K가 작은 K보다 먼저 실행되는 경우도 포함한다.

**채택 판단:** CPU 10% 이상 개선이 우선 목표다. reusable workspace가 peak RSS를
늘린다면 배열 수명 또는 상한을 조정하고, 3% 초과 RSS 회귀 시 채택하지 않는다.

### C-02 — PCA 자동 차원 탐색의 projection 재사용

**목표:** PCA 차원 후보별로 동일한 중심화/투영을 반복하는 경로가 있다면, 수학적으로
동등한 단일 계산과 prefix slice로 교체해 CPU와 임시 배열 RSS를 줄인다.

**조사 범위:** `pca_dimension_search` 및 `fit_clustering_pca`에서 후보 차원마다 PCA
fit, transform, normalization, neighbor/metric 계산이 어떤 순서로 반복되는지 확인한다.

**설계 단계:**

1. PCA solver·random state·whitening·정규화 조건을 확인하고, 한 번의 최대 차원
   projection을 앞 열 slice로 재사용해도 동등한 경우만 대상으로 한다.
2. 후보별 점수가 projection만 다르고 후속 거리 계산도 동일하다면, 최대 projection을
   한 번 만들고 필요한 열 view를 넘긴다.
3. view가 오래 살아 peak RSS를 키우지 않도록 탐색 종료 시 최대 projection과
   candidate metric artifact를 해제한다.
4. solver 특성상 prefix 동등성이 보장되지 않거나 candidate마다 fit이 의도된 경우,
   fit 재사용은 하지 않고 후보별 중복 후처리만 줄인다.

**검증:** 후보별 score와 최종 선택 차원, PCA component/sign 허용 오차, downstream
K 선택·label·center를 비교한다. projection 재사용 전후에 같은 seed에서 neighbor
관계와 클러스터 품질이 달라지지 않아야 한다.

**채택 판단:** PCA 탐색이 R-00 CPU 상위 병목일 때만 구현한다. 1,000/3,000 rows에서
CPU/RSS 모두를 비교하고, 선택 차원 또는 최종 clustering 결과가 달라지면 원인을
설명하지 못하는 한 보류한다.

### M-01 — 증분 center contribution의 compact numeric 저장

**목표:** `center_contributions`/`center_statistics`의 중첩 dict·객체 구조를 줄여
증분 update CPU, peak RSS, state 크기를 함께 낮춘다.

**조사 범위:** contribution 생성·병합·대체 경로와 state payload에서 각 필드가 차지하는
바이트 및 객체 수를 측정한다. 수정 대상은 `incremental_clustering.py`의 update/state
경로와 관련 migration 코드다.

**설계 단계:**

1. path/level/cluster 및 record ID별로 필요한 값(가중 합, count, vector 등)을
   명시하고, 실제로 per-record 복원이 필요한지 확인한다.
2. per-ID dict가 필요하면 안정적인 ID index + 연속 numeric array 형태를 설계한다.
   순회 순서나 pickle 구현 세부에 의존하지 않는다.
3. replace와 재실행 idempotency가 기존과 정확히 같도록, 이전 contribution을 빼고
   새 contribution을 더하는 경로를 새 구조에도 제공한다.
4. 새 state version 또는 명시적 migration을 추가한다. legacy state는 읽을 수 있어야
   하며, 한 번 저장하면 새 형식으로 정규화할 수 있다.
5. 큰 임시 Python 객체를 만들지 않는 vectorized 병합을 우선하되, 작은 batch에서
   변환 비용이 더 크면 threshold 또는 단순 경로를 둔다.

**검증:** append, replace, 같은 입력 재실행, 여러 path/level, save-load 후 update를
기준 구현과 비교한다. center/통계/점수/K/label 동등성과 state 크기, update p50,
peak RSS를 기록한다.

**채택 판단:** 3,000 rows에서 state 또는 RSS 10% 이상 감소하면서 update CPU가
악화되지 않아야 한다. state 구조를 바꾸는 만큼 migration과 corrupt-state 오류도
테스트해야 한다.

### I-01 (N-07) — state envelope 직렬화 중복 제거

**목표:** state 저장·로드 중 payload를 여러 번 pickle로 materialize하는 비용을 줄여
state I/O의 CPU와 일시 RSS를 낮춘다.

**전제:** 3,000 rows state가 약 47 MB이므로, 이 작업은 `save_state`/`load_state`가
실제 운영 경로에서 의미 있는 비중인지 R-00 또는 별도 I/O 프로파일로 확인한 뒤
진행한다. fit CPU가 주 병목이면 C-01~M-01보다 뒤에 둔다.

**설계 단계:**

1. `checked_state_envelope`, `atomic_pickle_dump`, `save_state`, `load_state`의
   payload 복사 수·최대 버퍼 크기·checksum 범위를 측정한다.
2. atomic write와 checksum 검증을 유지한 채 streaming checksum 또는 manifest/sidecar
   방식을 검토한다. 쓰는 도중의 state가 정상 파일처럼 보이면 안 된다.
3. 단일 파일 envelope을 유지할 수 있는 방안과 versioned migration 비용을 비교한다.
   format 변경은 legacy read 호환 및 명확한 오류 메시지를 필수로 한다.
4. checkpoint가 잦은 경우에만 compression을 별도 실험한다. compression은 CPU와 RSS
   동시 목표에 반할 수 있으므로 기본 채택안으로 가정하지 않는다.

**검증:** 정상 round-trip, 이전 버전 state load, checksum 불일치, 부분 파일/atomic
failure, large-state save/load 시간과 peak RSS를 확인한다.

**채택 판단:** load/save p50 또는 해당 구간 peak RSS가 10% 이상 개선되고 fit 결과와
state 호환성이 유지될 때만 채택한다.

### M-02 — level soft-membership 출력의 필요 시 생성

**목표:** N-06의 기본 비활성화 이후에도 남아 있는 `soft_memberships_by_level` 및
DataFrame `level_*_membership_*` 생성 비용이 큰지 확인하고, 필요할 때만 만든다.

**전제:** 현재 증거상 전체 CPU 개선은 작다. 따라서 field-level 메모리 프로파일에서
대상 출력이 material한 경우에만 수행한다.

**설계 단계:**

1. visualization/export/API 소비자가 요구하는 membership field와 fallback을
   목록화한다.
2. 기본 결과에는 path/label 중심의 최소 정보만 보관하고, soft membership은 명시
   옵션 또는 실제 소비 지점에서 계산한다.
3. 옵션 on/off가 예측 가능한 schema를 갖도록 하고, 없어진 열을 전제로 하는
   코드에는 명확한 오류 또는 계산 fallback을 둔다.

**검증:** 옵션 off에서 RSS/state/CPU, 옵션 on에서 기존 시각화·export 결과, schema
및 테스트를 확인한다.

**채택 판단:** 기본 경로 RSS/state가 10% 이상 줄거나 실제 소비 없는 대형 출력이
확인될 때만 추가 축소한다. CPU 개선 기대치는 낮으므로 C-01/C-02보다 앞서지 않는다.

### H-01 (N-04) — worker 병렬화 재검토 (보류)

**현 상태:** 2 CPU 측정에서 Python worker 병렬화는 느렸다. process 생성, pickle,
메모리 복제, BLAS oversubscription 비용이 이득을 상쇄한 것으로 본다.

**재개 조건:** 4개 이상 실제 사용 가능한 CPU, 충분히 큰 sibling/restart 작업량,
worker당 BLAS thread 제한, 그리고 직렬 대비 CPU 병목이 명확한 경우다.

**검증:** 1/2/4 worker와 BLAS thread 조합을 분리 측정하고, wall time뿐 아니라
aggregate RSS, 결과 순서·seed 재현성, 오류 전파를 확인한다. peak RSS가 크게 늘면
CPU가 개선돼도 기본값으로 채택하지 않는다.

## 실행 순서와 의사결정

1. **R-00**을 먼저 완료하고, 프로파일 표로 C-01과 C-02 중 실제 상위 병목을 확정한다.
2. CPU 상위인 **C-01** 또는 **C-02**를 한 번에 하나만 구현·검증한다. 두 변경을
   합치지 않아 효과를 분리할 수 없게 만들지 않는다.
3. 다음으로 **M-01**을 수행해 update path와 state/RSS를 함께 개선한다.
4. state I/O 비중이 확인되면 **I-01(N-07)**, 큰 soft-membership 출력이 확인되면
   **M-02**를 진행한다.
5. **H-01(N-04)**은 재개 조건이 충족될 때까지 보류한다.

각 작업이 끝나면 이 문서와
[PYTHON_PERFORMANCE_BACKLOG.md](PYTHON_PERFORMANCE_BACKLOG.md)의 결과 표에 기준값,
후보값, 변화율, 사용한 명령, 결론(채택/보류/폐기)을 기록한다. 성능 개선이 충분하지
않아도 측정값과 보류 사유를 남겨 같은 실험을 반복하지 않게 한다.
