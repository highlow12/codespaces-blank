# Atomic Clusters 제품화 계획

기준일: 2026-09-05

이 문서는 Atomic Clusters MVP를 실제로 계속 사용할 수 있는 제품으로 다듬기 위한 구현 계획이다. 현재 핵심 흐름인 `Vault → embedding → clustering → hierarchy → keyword titles → Explorer`는 동작하므로, 이후 개발은 새로운 연구 기능을 넓히기보다 사용 중 반복해서 발생하는 비용과 실패 지점을 줄이는 데 집중한다.

## 0. 목표와 원칙

### 제품 목표

사용자가 Obsidian 노트를 폴더로 직접 재분류하지 않아도, Atomic Clusters가 변경된 노트를 지속적으로 반영하고, 자동으로 만든 의미 구조를 빠르게 탐색하며, 잘못된 자동 판단은 사람이 쉽게 교정할 수 있게 한다.

핵심 사용 경험은 다음과 같다.

```text
노트를 평소처럼 작성한다
  → 변경분만 자동으로 감지한다
  → 필요한 embedding만 갱신한다
  → 가능한 범위에서 기존 구조를 재사용한다
  → Explorer에서 바로 결과를 본다
  → 검색과 필터로 원하는 영역을 좁힌다
  → 잘못된 제목·배치가 있으면 사람이 수정한다
  → 다음 재빌드에서도 사용자의 수정은 보존된다
```

## 0.1 2026-09-03 구현 상태 — 이전 remaining-work 목록의 #2, #3, #5

현재 소스와 실제 release-WASM 측정 결과를 기준으로 한 상태 기록이다. 아래의
`구현됨`은 해당 기능의 구현 상태를 뜻하며, 대규모 성능 항목은 측정 한계까지
별도로 판정한다.

- **#2 — per-note/folder exclusion (plan §2.6): 구현됨.** note context
  menu와 folder context menu에서 Atomic Clusters 대상 제외/복구를 하고,
  설정에 vault-relative exclusion을 저장한다. 2026-09-03 당시에는 수동
  cluster title 수정/reset, note preference, manual group, feedback log가
  미구현이었으며, 해당 기능은 아래 `0.3`의 Milestone C에서 후속 구현했다.
  상위 폴더 상속 메뉴와 마지막 활성 노트 제외 처리도 2026-09-04에 수정했고,
  context menu·rename·automatic refresh 경계의 회귀 테스트를 추가했다.
- **#3 — Note detail panel (plan §3.6): 구현됨.** 선택한 note의 path/title,
  automatic leaf, hierarchy, membership/probability와 관련 note 정보를
  확인하고 원본 note를 여는 detail panel을 구현했다. Search & Focus는 이전에
  구현되어 있던 범위이며 이번 번호 작업의 구현 완료로 다시 주장하지 않는다.
  manual preferred cluster/note preference는 2026-09-05 Milestone C 통합에서
  구현했다.
- **작업 5 — Large Vault hardening: 부분 완료.** 계측 runner, renderer memory
  preflight, persistent Notice heartbeat, cancellation/progress/RSS 계측과
  release WASM 검증을 구현했다. 2026-09-03 실제 Gemini 3,000-record 입력의
  release-WASM 실행은 1,000행 `97,745 ms`/peak RSS `1,218,125,824 bytes`,
  3,000행 `167,310 ms`/peak RSS `1,623,543,808 bytes`였고, 두 실행 모두
  반복적인 250 ms 초과 main-thread stall은 관찰되지 않았다. cancellation은
  137–219 ms, release asset/export 검증은 통과했다.
  - 5,000/10,000행은 현재 실제 입력이 3,000행뿐이므로 중복·외삽 없이
    `unavailable`로 남긴다.
  - PCA → UMAP progress 침묵 구간은 1,000행 `88,648 ms`, 3,000행
    `144,482 ms`로 측정됐다. heartbeat는 이 동안 Notice의 elapsed/“Still
    working” 가시성을 보완하지만 알고리즘 progress gap을 없애지 않으므로,
    이 수치와 5,000/10,000 미측정 상태 때문에 작업 5의 제품 acceptance를
    완료로 표시하지 않는다.

상세 raw 결과는 `/tmp/atomic-clusters-large-vault-hardening-final-2026-09-03/`
및 `atomic-clusters/docs/large-vault-hardening-report.md`에 있다.

## 0.2 리뷰 후속 작업 — exclusion 경계 조건 (완료)

2026-09-03 코드 리뷰에서 확인한 두 경계 조건은 2026-09-04 수정과 회귀
테스트를 완료했다. 아래 항목은 유지해야 할 acceptance 조건으로 남긴다.

- **상위 폴더 제외 상속을 파일 메뉴에 반영한다.** 노트가 `excludedFolders`의
  하위에 있으면 이미 제외된 상태이므로 파일 메뉴에 중복 `Exclude`를 제공하지
  않는다. 직접 `excludedNotes` 항목이 있어도 `Restore`가 상위 폴더 규칙을
  무시하는 것처럼 보이면 안 된다. 메뉴를 비활성화하거나 상속 상태를 명확히
  표시한다.
- **마지막 활성 노트 제외를 정상 처리한다.** 활성 노트가 0개가 되는 경우에도
  변경 queue를 오류로 되돌리지 않는다. 빈 structural result로 교체하거나,
  마지막 노트 제외를 명시적으로 막고 사용자에게 이유를 안내해야 한다. 설정상
  제외된 노트가 이전 Explorer 결과에 계속 표시되어서는 안 된다.
- 두 경우 모두 context menu, 설정 복구, rename, automatic refresh on/off 상태를
  회귀 테스트한다.

## 0.3 2026-09-05 구현 상태

현재 소스와 전체 회귀 테스트를 기준으로 제품화 마일스톤을 다시 판정한다.

- **Milestone A — Explorer 신뢰성: 구현됨.** pan camera constraint를 interaction
  중 적용하고, label bounding box·hover radius·resize를 반영한 safe padding과
  pointerleave/pointercancel/drag hover clear를 구현했다. 빠른 drag 20회와 여러
  zoom, resize, 빈 공간 hover 회귀 테스트를 추가했다.
- **Milestone B — Search & Focus: 구현됨.** 상세 acceptance 상태는
  `atomic-clusters/docs/search-focus-plan.md`에 기록되어 있다.
- **Milestone C — Manual corrections: 구현됨.** stable member fingerprint,
  Jaccard 0.7 기반의 보수적 rebuild 승계, manual title rename/reset, note
  preference, view-level manual group/ungroup, too-broad feedback와 로컬 feedback
  log를 SQLite에 저장한다. Explorer와 note context menu에서 수정할 수 있고,
  Search의 `Manually adjusted` 필터와 note detail이 실제 저장 상태를 사용한다.
  generated title과 manual title은 별도 데이터로 유지한다.
- **Milestone D — Automatic incremental refresh: 구현됨.** Vault event queue,
  debounce, create/modify/delete/rename, changed-only embedding, no-op/soft/full
  정책, provisional placement, 자동 refresh 설정, atomic result publication과
  재시작 복구가 구현되어 있다. rename 시 manual correction의 note path와
  stable cluster reference도 함께 보존한다.
- **Milestone E — Large Vault hardening: 부분 완료 유지.** memory preflight,
  heartbeat, cancellation 경계, release-WASM 검증과 TypeScript fallback 제한을
  구현했다. 동기 PCA/HDBSCAN 구간 전후에는 event-loop yield와 cancellation
  check를 수행하지만 연산 도중 취소 가능성을 과장하지 않는다. 실제 데이터가
  3,000행뿐이므로 5,000/10,000 실측은 계속 `unavailable`이다.

2026-09-05 검증 결과는 plugin Node test 27/27, repository Python test 132/132,
TypeScript `--noEmit`, plugin build 통과다. Gemini 3,000-record 입력의 고정
seed 100행 fast smoke fit도 성공했다. 실제 Obsidian에서 manual-correction
interaction과 layout을 확인하는 수동 smoke test는 별도로 남아 있다.

### 제품화 원칙

1. **자동화보다 통제 가능성이 우선이다.** 자동 결과는 수정할 수 있어야 한다.
2. **재계산보다 재사용이 우선이다.** content hash, embedding cache, PCA 상태, cluster 결과를 가능한 한 재사용한다.
3. **사용자 수정은 파생 결과보다 강하다.** 자동 제목이나 자동 배치는 재생성될 수 있지만 manual override는 명시적으로 지우기 전까지 보존한다.
4. **구조와 시각화를 분리한다.** 검색, 필터, pan, zoom, hover 같은 표시 상태가 clustering 결과를 변경하지 않는다.
5. **불확실성을 숨기지 않는다.** noise, 낮은 membership, 임시 배치, 오래된 결과를 UI에서 구분한다.
6. **대규모 Vault에서는 즉시성보다 중단 가능성과 진행 가시성이 중요하다.** 오래 걸리는 작업은 언제든 취소할 수 있고, 어디까지 진행됐는지 보여야 한다.

---

# 1. 부분 업데이트와 자동 재정리

## 1.1 문제

현재 embedding cache는 note content hash를 사용하므로 수정되지 않은 노트의 임베딩은 재사용할 수 있다. 그러나 clustering은 여전히 사용자가 `Build note clusters`를 실행하여 전체 구조를 다시 만드는 흐름이 중심이다.

실제 사용에서는 다음 상황이 반복된다.

- 기존 Vault에 노트 2개 추가
- 노트 1개 수정
- 파일 이름 변경
- 노트 삭제
- 하루 동안 수십 번 저장

이때 매번 전체 UMAP/HDBSCAN을 다시 돌리면 Vault가 커질수록 자동정리 도구가 오히려 사용자의 시간을 요구하게 된다.

## 1.2 목표

변경된 노트를 자동으로 감지하고, 변경 범위와 구조 드리프트가 작으면 기존 결과를 최대한 재사용한다. 전체 재클러스터링은 필요한 경우에만 수행한다.

첫 제품 버전에서는 완전한 online HDBSCAN을 구현하는 것을 목표로 하지 않는다. 대신 **빠른 변경 감지 + debounce + 재사용 + 재빌드 정책**으로 체감 비용을 크게 줄이고, 이후 실제 측정 결과를 보고 더 깊은 증분 clustering을 도입한다.

## 1.3 변경 감지

Obsidian Vault 이벤트를 사용한다.

대상 이벤트:

- `create`: 새 Markdown 노트
- `modify`: 기존 노트 내용 변경
- `delete`: 노트 삭제
- `rename`: 경로 변경

이벤트가 들어올 때 즉시 clustering을 실행하지 않는다. 다음 구조의 pending queue를 둔다.

```ts
interface PendingVaultChanges {
  created: Set<string>;
  modified: Set<string>;
  deleted: Set<string>;
  renamed: Map<string, string>; // oldPath -> newPath
  firstChangedAt: number;
  lastChangedAt: number;
}
```

### Debounce 정책

기본값:

- 마지막 변경 후 5초 동안 추가 변경이 없으면 background refresh 후보가 된다.
- 사용자가 계속 편집 중이면 최대 60초까지 연기한다.
- clustering 작업이 이미 실행 중이면 현재 작업을 자동 취소하지 않고 pending queue에 누적한다.
- 작업 종료 후 pending 변경이 남아 있으면 한 번만 후속 refresh를 실행한다.

설정 옵션:

- `Automatic refresh`: On / Off
- `Refresh delay`: 기본 5초
- `Only refresh when Obsidian is idle`: 추후 옵션

## 1.4 업데이트 단계

### 단계 A — 메타데이터와 embedding 증분 갱신

1. 변경된 path만 다시 읽는다.
2. content hash를 비교한다.
3. hash가 바뀐 노트만 embedding provider로 보낸다.
4. rename인데 content hash가 같으면 기존 embedding record의 path만 이전한다.
5. delete된 노트는 active note 집합에서 제거한다.
6. 실패한 노트는 마지막 정상 embedding을 조용히 재사용하지 않는다. UI에서 `embedding stale/failed` 상태로 표시한다.

SQLite에는 최소 다음 정보가 있어야 한다.

```text
notes
- path
- content_hash
- mtime
- active

embeddings
- provider
- model
- path
- content_hash
- vector
- updated_at
```

rename은 가능하면 vector 복사를 하지 않고 key metadata만 변경한다.

### 단계 B — 구조 갱신 모드 결정

변경량에 따라 세 모드를 둔다.

#### 1) No-op

다음 조건에서는 hierarchy를 다시 만들지 않는다.

- rename만 있었고 모든 content hash가 동일
- 제외 폴더 설정 변경이 없음
- embedding provider/model 변경이 없음

이 경우 ID/path 참조와 UI만 갱신한다.

#### 2) Soft refresh

초기 정책은 전체 노트의 변경 비율이 충분히 작을 때 사용한다.

예시 기본 조건:

```text
changed active notes <= max(20, 전체의 2%)
AND deleted notes <= 전체의 1%
AND embedding model unchanged
AND PCA model available
```

Soft refresh의 1차 구현은 다음과 같다.

1. 저장된 PCA model로 변경 노트를 projection한다.
2. 기존 leaf center와 cosine similarity를 계산한다.
3. 새/수정 노트를 가장 가까운 leaf 후보에 임시 배치한다.
4. 낮은 similarity 또는 높은 outlier 조건이면 `unsettled`로 둔다.
5. Explorer는 즉시 업데이트된 임시 결과를 보여준다.
6. 구조 품질 지표 또는 누적 변경량이 threshold를 넘으면 background full rebuild를 예약한다.

중요: 이것은 HDBSCAN을 증분 학습했다고 주장하지 않는다. UI와 저장 구조에서 `provisional placement`임을 명확히 구분한다.

#### 3) Full rebuild

다음 조건에서는 전체 구조를 재계산한다.

- embedding provider/model 변경
- PCA model 호환 불가
- 변경량 threshold 초과
- 삭제량이 큼
- provisional 노트 비율이 높음
- 기존 leaf occupancy가 크게 변함
- 사용자가 `Rebuild all clusters`를 명시적으로 실행

## 1.5 드리프트 기준

초기에는 단순하고 설명 가능한 기준을 사용하고, 실제 Vault 로그를 바탕으로 보정한다.

기록할 지표:

- 전체 active note 수
- 변경 note 비율
- provisional note 비율
- noise 비율 변화
- leaf별 occupancy 변화
- 신규 노트의 nearest-center distance 분포
- 마지막 full rebuild 이후 누적 변경 비율

초기 full rebuild 후보:

```text
누적 변경 >= 전체의 10%
OR provisional >= 전체의 5%
OR 특정 leaf occupancy 변화가 이전 대비 30% 이상
OR 마지막 full rebuild 후 7일 이상 + 변경 존재
```

시간 기준은 강제 정책이 아니라 기본 heuristic으로 두고 설정에서 자동 rebuild를 끌 수 있게 한다.

## 1.6 UI

Explorer 상단에 구조 상태를 표시한다.

```text
Up to date
3 notes pending
12 provisional placements
Rebuild recommended
Building 42%
```

필요한 명령:

- `Refresh changed notes`
- `Rebuild all clusters`
- `Pause automatic refresh`

진행 중 편집이 발생해도 현재 결과는 유지하고 완료 후 다음 변경분을 처리한다.

## 1.7 저장과 복구

refresh는 단계별 atomic transaction으로 처리한다.

1. note metadata 갱신
2. embedding 갱신
3. provisional assignment 또는 새 structural result 저장
4. successful result pointer 교체

중간 실패 시 이전 structural result는 계속 열 수 있어야 한다.

작업 시작 시 이전 결과를 삭제하지 않는다.

## 1.8 테스트

필수 테스트:

- modify 1개 → 해당 노트만 재임베딩
- 동일 내용 rename → 재임베딩 없음
- delete → 결과에서 제거
- create/modify 폭주 → debounce 후 1회 refresh
- refresh 중 추가 수정 → 후속 queue로 이동
- embedding 일부 실패 → 기존 structural result 보존
- soft refresh → provisional flag 저장/복구
- threshold 초과 → full rebuild 예약
- plugin restart → pending이 아닌 마지막 성공 상태 복구

## 1.9 완료 기준

- 노트 한두 개 수정 시 전체 embedding을 다시 계산하지 않는다.
- rename만으로 expensive clustering을 실행하지 않는다.
- 자동 refresh를 완전히 끌 수 있다.
- 임시 배치와 정식 clustering 결과가 UI에서 구분된다.
- 어떤 실패가 발생해도 마지막 성공 Explorer 결과를 잃지 않는다.

---

# 2. 사용자 수정과 피드백

## 2.1 문제

자동 clustering과 자동 keyword title은 반드시 일부 잘못된 결과를 만든다. 제품이 신뢰를 얻으려면 알고리즘이 완벽해야 하는 것이 아니라, 사용자가 잘못된 부분을 빠르게 수정할 수 있어야 한다.

사용자의 수정은 두 종류로 나눈다.

1. **표시 수정**: cluster 이름, 색상, 숨김 등 구조를 바꾸지 않는 것
2. **구조 수정**: note의 선호 cluster, merge, split 등 의미 구조에 영향을 주는 것

첫 구현은 표시 수정과 가벼운 구조 힌트부터 시작한다.

## 2.2 수동 cluster title override

가장 먼저 구현한다.

Cluster Explorer의 cluster 제목에 context menu 또는 edit action을 제공한다.

기능:

- `Rename cluster`
- `Reset to generated title`
- generated title을 tooltip 또는 secondary text로 확인

저장 모델:

```ts
interface ClusterTitleOverride {
  stableClusterKey: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}
```

### stable cluster key

node id는 rebuild마다 바뀔 수 있으므로 그대로 영구 key로 쓰면 안 된다.

초기 key는 cluster 구성원의 안정적인 fingerprint로 만든다.

```text
hash(sorted(member note paths 또는 note identity ids))
```

rebuild 뒤 정확히 동일한 집합이 아니더라도 기존 cluster와 높은 overlap이 있으면 override를 승계할 수 있다.

승계 기준 baseline:

- Jaccard overlap 0.7 이상
- 후보가 하나만 명확히 우세할 때만 자동 승계
- 애매하면 override를 orphan 상태로 두고 사용자에게 강제로 적용하지 않음

## 2.3 노트별 cluster 선호 표시

사용자가 노트를 보고 `이 노트는 여기가 더 맞다`고 피드백할 수 있게 한다.

첫 버전에서는 hard move로 clustering 결과 자체를 변경하지 않는다.

UI:

- note context menu → `Prefer another cluster…`
- 후보 cluster는 현재 hierarchy에서 가까운 상위 5개를 제시
- 선택 후 Explorer에서 해당 연결을 `user preferred`로 표시

저장:

```ts
interface NoteClusterPreference {
  notePath: string;
  preferredClusterKey: string;
  createdAt: string;
}
```

사용 목적:

- 표시 우선순위
- 검색/필터
- 향후 reranking 또는 semi-supervised clustering 실험 데이터

중요: 초기 버전에서는 사용자의 클릭 한 번으로 HDBSCAN label을 조작하지 않는다. 원본 자동 결과와 사용자 선호를 별도 채널로 보존한다.

## 2.4 cluster merge

두 cluster가 사실상 같은 주제라면 사용자가 논리적으로 합칠 수 있게 한다.

첫 구현은 **view-level manual group**으로 처리한다.

예:

```text
자동 cluster A: Unity Shader
자동 cluster B: URP Rendering

사용자 manual group: Unity Rendering
  ├─ A
  └─ B
```

이 방식은 원본 hierarchy를 파괴하지 않고도 원하는 정리를 제공한다.

기능:

- cluster 두 개 이상 선택
- `Group clusters`
- manual group title 입력
- ungroup 가능

이 manual group은 다음 rebuild에서도 child cluster overlap을 기반으로 최대한 복원한다.

## 2.5 split 기능

직접적인 cluster split은 비용과 모호성이 크므로 1차 제품화 범위에서는 보류한다.

대신 다음 UX를 제공한다.

- `This cluster is too broad` 피드백 기록
- 현재 hierarchy 안에 하위 node가 있으면 한 단계 깊게 보기
- flat leaf가 너무 넓으면 feedback event를 저장

향후 split hint를 HDBSCAN `minClusterSize` 또는 local sub-clustering 실험에 사용할 수 있다.

## 2.6 제외 기능

사용자가 특정 노트나 폴더를 clustering 대상에서 제외할 수 있어야 한다.

현재 excluded folders 설정을 확장한다.

추가 기능:

- note context menu → `Exclude from Atomic Clusters`
- folder context menu → `Exclude folder`
- Settings에서 목록 확인 및 복구

노트 제외는 원본 Markdown을 수정하지 않고 plugin 설정/DB에서 관리한다.

선택적으로 frontmatter 기반 opt-out은 후속 기능으로 고려한다.

```yaml
atomic-clusters: false
```

하지만 plugin이 사용자 노트에 frontmatter를 자동 삽입하지 않는다.

## 2.7 수정 결과의 우선순위

표시 시 우선순위:

```text
manual title override
> generated keyword title
> representative note fallback
```

cluster 관계 표시:

```text
manual group / preferred placement
+ original automatic hierarchy
```

원본 자동 결과를 완전히 덮어쓰지 않아야 `Reset`, 비교, 재생성이 가능하다.

## 2.8 피드백 로그

사용자 행동을 외부로 전송하지 않고 로컬 SQLite에 저장한다.

예시 이벤트:

- title renamed
- title reset
- note preferred cluster changed
- manual group created/deleted
- cluster marked too broad
- note excluded/restored

이 로그는 향후 알고리즘 개선용 사용자 본인 데이터로 쓸 수 있지만 자동 학습에는 사용하지 않는다.

## 2.9 테스트

- manual title이 title regeneration 후에도 유지
- reset 시 최신 generated title 복원
- rebuild 후 동일/유사 cluster에 override 승계
- ambiguous overlap에서는 잘못 승계하지 않음
- manual group 생성/해제
- note preference 저장/복구
- note 제외 후 embedding/clustering 대상에서 제거
- plugin restart 후 모든 override 유지

## 2.10 완료 기준

- 사용자는 자동 제목이 틀리면 2회 이하의 interaction으로 수정할 수 있다.
- 제목 재생성이나 전체 rebuild가 manual title을 임의로 지우지 않는다.
- 사용자는 특정 노트를 제외할 수 있다.
- 자동 결과와 사용자 수정이 데이터 모델상 분리되어 언제든 원복 가능하다.

---

# 3. 검색과 필터

## 3.1 문제

Cluster Explorer가 시각적으로 유용해도 노트가 수천 개가 되면 화면을 직접 훑는 것만으로 원하는 정보를 찾기 어렵다. 자동정리 결과를 실제 탐색 도구로 만들려면 검색과 필터가 필요하다.

검색의 1차 목표는 새로운 RAG 시스템을 만드는 것이 아니다. 이미 생성된 note metadata, cluster title, keyword, hierarchy를 빠르게 좁혀 보여 주는 것이다.

## 3.2 Explorer search bar

Explorer 상단에 항상 보이는 검색창을 추가한다.

검색 대상:

1. note title
2. note path
3. note body의 lexical match
4. generated/manual cluster title
5. title generation keyword
6. Obsidian tag
7. alias

초기 구현은 로컬 lexical search로 충분하다.

### 검색 결과 표현

검색 중 지도 전체를 제거하지 않는다.

- matching note: 강하게 강조
- matching cluster: 강조
- 비매칭 note: opacity 감소
- 검색 결과가 속한 ancestor hierarchy는 유지

검색 결과 패널에는 다음을 보여 준다.

```text
12 notes · 3 clusters

Clusters
- Embedding · Clustering (7)
- Unity Rendering (3)

Notes
- 임베딩 공간 정렬
- HDBSCAN 실험
...
```

## 3.3 검색 syntax

처음부터 복잡한 query language는 만들지 않는다.

지원 baseline:

```text
embedding
"machine learning"
tag:#unity
path:Projects/
cluster:rendering
```

후속 후보:

```text
is:noise
is:provisional
is:manual
membership:>0.7
```

## 3.4 필터

검색과 별도의 quick filter chips를 제공한다.

기본 필터:

- `All`
- `Current cluster`
- `Noise`
- `Provisional`
- `Manually adjusted`
- `Recently changed`

고급 필터:

- folder
- tag
- cluster depth
- membership/probability range
- embedding provider/model

필터는 조합 가능해야 한다.

예:

```text
folder=Projects/Game
AND tag=#unity
AND probability >= 0.7
```

## 3.5 cluster focus mode

cluster를 선택하면 `Focus` 명령을 제공한다.

Focus mode에서는:

- 해당 cluster의 subtree만 표시
- breadcrumb로 상위 hierarchy 이동
- sibling cluster로 빠르게 이동
- 검색은 focus 범위 안에서 우선 수행

예:

```text
All notes > Programming > Game Development > Unity
```

`Esc` 또는 breadcrumb root 선택으로 전체 지도로 돌아간다.

## 3.6 note detail panel

검색/필터와 함께 note detail을 강화한다.

선택한 note에 표시:

- title/path
- 현재 automatic leaf
- ancestor hierarchy
- membership/probability
- noise/provisional 여부
- manual preferred cluster 여부
- 가까운 관련 노트 상위 N개
- 관련 cluster title keywords

`Open note` 버튼으로 원본 Markdown을 연다.

이 패널은 향후 `왜 여기 있나?` 설명 기능의 기반으로 사용한다.

## 3.7 인덱싱

첫 버전에서 별도 검색 엔진을 추가하지 않는다.

Vault scan 시 SQLite에 searchable text metadata를 저장하거나 현재 NoteRecord를 이용한 in-memory index를 구성한다.

Vault가 커졌을 때 다음 순서로 확장한다.

1. title/path/tag/alias in-memory inverted index
2. SQLite FTS 사용 검토
3. lexical search가 실제 병목일 때만 별도 index 도입

embedding similarity search는 별도 기능으로 분리한다. 검색창에 처음부터 semantic search를 섞으면 결과 설명과 성능 특성이 복잡해진다.

## 3.8 키보드 UX

- `/` 또는 `Cmd/Ctrl + F`: Explorer search focus
- `Esc`: 검색 초기화 또는 focus mode 한 단계 나가기
- `Enter`: 첫 결과 열기
- 화살표: 결과 목록 이동

Obsidian 기본 단축키와 충돌하면 command 등록 방식으로 조정한다.

## 3.9 테스트

- title/path/body 검색
- 한글/영문 혼합 검색
- phrase 검색
- tag/path/cluster qualifier
- 검색 + filter 조합
- focus mode + 검색
- 검색 중 hierarchy가 사라지지 않음
- 결과 클릭 → 정확한 note open
- 10,000개 metadata fixture에서 입력 지연 측정

## 3.10 완료 기준

- 사용자는 지도를 직접 훑지 않고 원하는 note/cluster를 검색할 수 있다.
- 검색해도 전체 의미 구조에서 결과가 어디에 있는지 볼 수 있다.
- 검색/필터는 clustering을 다시 실행하지 않는다.
- 10,000개 note metadata에서도 일반적인 타이핑이 끊기지 않는다.

---

# 4. 기존 README에서 이관한 Explorer UI/UX 과제

현재 `atomic-clusters/README.md`에 기록된 visualization 문제를 이 계획의 정식 제품화 항목으로 편입한다.

## 4.1 Pan 중 overshoot / snapback

현재 viewport panning 중 note cloud가 viewport보다 멀리 이동했다가 다시 돌아오는 현상이 있다.

목표:

- drag 동안과 drag 종료 후 transform 계산을 동일 좌표계로 통일
- viewport constraint를 animation 종료 시점이 아니라 interaction 중 지속 적용
- 사용자 입력과 자동 recenter가 동시에 위치를 변경하지 않게 상태 분리

완료 기준:

- 빠른 drag를 20회 반복해도 snapback이 보이지 않음
- zoom level이 달라도 동일

## 4.2 Edge clipping

viewport 경계의 note와 label이 잘리는 문제를 수정한다.

목표:

- point가 아니라 렌더링 bounding box 기준으로 safe padding 계산
- label 크기와 hover 확대를 고려한 여백 확보
- resize 시 bounds 재계산

완료 기준:

- 기본 zoom에서 화면 가장자리 cluster title/note label이 잘리지 않음
- 창 크기 변경 후에도 유지

## 4.3 Hover 잔류

빈 공간으로 pointer가 이동해도 이전 hover가 남는 문제를 수정한다.

목표:

- hover target을 hit-test 결과에서 단일 source of truth로 관리
- pointerleave/pointercancel 처리
- 빈 공간 hit 시 즉시 hover clear
- 선택 상태와 hover 상태를 분리

완료 기준:

- 빈 공간에서는 highlight/tooltip이 남지 않음
- drag 중 hover가 잘못 고정되지 않음

---

# 5. 기존 개발 현황 문서에서 이관한 기술 과제

## 5.1 3,000개 이상 Vault 성능·메모리·취소 UX

현재 full 경로는 3,000행에서 수 분이 걸릴 수 있다. 따라서 제품화 과정에서 반드시 큰 Vault를 대상으로 다시 측정한다.

측정 규모:

- 1,000
- 3,000
- 5,000
- 10,000

기록:

- 전체 wall time
- embedding 제외 clustering time
- PCA
- UMAP
- HDBSCAN/MST
- hierarchy
- title generation
- peak memory
- worker responsiveness
- cancellation latency

목표:

- 250ms 이상 UI main-thread stall이 반복되지 않음
- cancel 요청 후 가능한 한 짧은 단계 경계에서 중단
- 장시간 작업에서도 진행 단계가 정지한 것처럼 보이지 않음
- 메모리 부족 가능성을 preflight 또는 실행 중 감지하여 안전한 오류 제공

## 5.2 WASM release 경로 강제 검증

큰 Vault에서 deterministic TypeScript fallback을 production 품질로 오해하지 않게 한다.

릴리스 빌드 정책:

- generated WASM asset 없으면 `build:release` 실패 유지
- Explorer diagnostics에 runtime backend 표시 가능하게 준비
- 큰 Vault에서 fallback이 선택된 경우 경고

## 5.3 HDBSCAN provider / Python parity

현재 `umap-js`와 `umap-learn`, Rust condensed-tree implementation과 Python native HDBSCAN은 완전 동일한 수치 결과를 보장하지 않는다.

제품 목표는 완전한 label parity가 아니라 다음 계약을 만족하는 것이다.

- 동일 입력/설정에서 재현 가능한 plugin 결과
- noise/probability/outlier가 제품 UX에 사용할 만큼 안정적
- provider 교체 시 schema와 hierarchy contract 유지

외부 `hdbscan-rs` 또는 별도 provider vendor는 **실제 성능/정확도 문제가 측정으로 확인될 때만** 진행한다. 단순히 reference와 다르다는 이유만으로 의존성을 늘리지 않는다.

---

# 6. 구현 순서

## Milestone A — Explorer 신뢰성

먼저 현재 사용을 방해하는 UI 문제를 없앤다.

- pan overshoot/snapback
- edge clipping
- hover clear
- note detail panel의 최소 골격

이 단계에서는 clustering 알고리즘을 변경하지 않는다.

## Milestone B — Search & Focus

Implementation plan: [`atomic-clusters/docs/search-focus-plan.md`](atomic-clusters/docs/search-focus-plan.md)

- search bar
- title/path/tag/cluster 검색
- filter chips
- cluster focus mode
- 결과 패널
- 키보드 탐색

이 단계가 끝나면 Explorer가 단순 visualization이 아니라 실제 탐색 인터페이스가 된다.

## Milestone C — Manual corrections

- manual cluster title
- reset generated title
- note exclusion
- note preferred cluster
- manual cluster group
- override persistence/migration

이 단계가 끝나면 자동 결과가 틀려도 사용자가 시스템을 버리지 않고 고쳐 쓸 수 있다.

## Milestone D — Automatic incremental refresh

- Vault event queue
- debounce
- rename/delete handling
- changed-only embedding
- no-op refresh
- provisional soft refresh
- full rebuild threshold
- automatic refresh controls

이 단계가 끝나면 사용자가 평소 노트를 작성하는 것만으로 정리 상태가 따라온다.

## Milestone E — Large Vault hardening

- 1,000 / 3,000 / 5,000 / 10,000 benchmark
- memory profiling
- cancellation latency
- progress UX
- WASM backend validation
- 필요 시 HDBSCAN provider 최적화

---

# 7. 제품 수준 완료 기준

다음 조건을 만족하면 기능적으로 `MVP`에서 일상 사용 가능한 `1.0 product` 단계로 넘어간 것으로 본다.

1. 새 노트와 수정 노트가 자동으로 감지된다.
2. 작은 변경에서 불필요한 전체 embedding 재계산이 없다.
3. 전체 rebuild가 필요한 이유를 사용자에게 설명할 수 있다.
4. 자동 cluster title을 사용자가 수정하고 영구 보존할 수 있다.
5. 원치 않는 노트/폴더를 제외할 수 있다.
6. 검색과 필터로 수천 개 노트에서 원하는 영역을 바로 찾을 수 있다.
7. cluster focus와 breadcrumb로 hierarchy를 이동할 수 있다.
8. pan/zoom/hover에서 눈에 띄는 상호작용 버그가 없다.
9. 장시간 clustering은 진행률과 취소를 제공하고 이전 성공 결과를 잃지 않는다.
10. 10,000개 규모 테스트에서 성능 한계와 권장 환경을 문서화한다.

---

# 8. 의도적으로 후순위로 두는 기능

다음 기능은 현재 제품 핵심 가치에 필요하지 않으므로 위 계획 완료 전에는 우선순위를 낮게 둔다.

- LLM 기반 cluster 요약
- RAG/채팅 인터페이스
- 자동 backlink 생성
- 이미지·음성·영상 멀티모달 입력
- 복잡한 자연어 query engine
- 클라우드 동기화 서비스
- 사용자 행동을 학습하는 자동 personalization

이 기능들은 `자동으로 정리된 지식 더미를 유지하고 탐색한다`는 핵심 경험이 안정된 뒤 별도 확장 과제로 평가한다.

---

# 9. 바로 다음 작업

가장 먼저 구현할 순서는 다음과 같다.

1. Explorer의 README visualization issue 3개 수정
2. Search bar + cluster focus mode
3. Manual cluster title override + reset
4. Note exclusion
5. Vault change watcher + changed-only embedding refresh
6. Provisional placement baseline
7. 3,000개 이상 benchmark 재측정

이 순서는 사용자가 즉시 체감할 수 있는 Explorer 품질과 통제 가능성을 먼저 확보하고, 그 뒤 자동 refresh로 상시 사용 비용을 낮추기 위한 것이다.
