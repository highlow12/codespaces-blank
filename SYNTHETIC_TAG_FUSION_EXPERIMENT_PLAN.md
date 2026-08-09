# 합성 데이터 기반 태그 융합 실험 계획서

## 1. 목적과 판단 기준

이 계획의 목적은 **태그가 유용한가**를 단일 점수로 판정하는 것이 아니라,
어떤 데이터 조건에서 어떤 융합 방식이 이득 또는 손해로 전환되는지를 밝히는
것이다. 현재 결과는 다음처럼 해석한다.

- 태그에는 의미 신호가 있다. 무작위 태그보다 올바른 태그의 결과가 좋다.
- 그러나 현재 품질의 태그를 본문 임베딩과 초기에 결합하면, 신호 이득보다
  표현 공간 왜곡 비용이 더 크다.
- 따라서 검증 전의 기본 설계는 `content → PCA → SFCM`을 유지하고, 태그는
  별도 metadata/prior/reranking 채널로 취급한다.

이 실험은 아래 네 질문에 답해야 한다.

1. 본문 임베딩의 SNR이 어느 수준에서 태그 융합이 content-only보다 유리해지는가?
2. 그 경계는 태그 corruption과 tag weight에 따라 어떻게 변하는가?
3. 관측된 손해는 fusion 자체, PCA 전처리, 또는 둘의 상호작용 중 어디에서 생기는가?
4. soft membership을 복원하는 SFCM의 장점이 경계 노트에서 실제로 나타나는가?

## 2. 범위와 비범위

포함 범위:

- correlated root를 가진 synthetic generator와 태그 관측 모델
- content noise × tag corruption × tag weight sweep
- fusion/PCA ablation, soft-membership 평가, K를 모르는 조건 평가
- 재현 가능한 CSV/JSON 결과와 phase diagram

제외 범위:

- 실제 Gemini 데이터에서 태그를 본문 공간에 즉시 추가하는 제품 변경
- 단일 synthetic 점수만으로 한 융합 방식을 일반화하는 결론
- 최종 metadata prior 또는 reranker의 제품 구현

## 3. Phase 0 — 측정 규약 고정

구현 전에 모든 run이 다음 메타데이터를 남기도록 규약을 고정한다.

- generator seed, SFCM seed, root 수, sample 수, embedding 차원
- content noise, 각 태그 corruption 확률, tag weight, fusion 방식, PCA 방식
- true K 사용 여부와 선택된 K
- 모든 hard/soft metric 및 경계 샘플 수

현재 코드의 `tag_embedding_features.build_tag_augmented_features`는
정규화한 content와 weighted tag를 **concatenation**한다. 이슈에서 지적한
`multi-tag → sum + normalize → content에 additive fusion`은 다른 연산이다.
보고서와 run 이름에는 반드시 둘을 별도의 실험군으로 기록한다.

## 4. Phase 1 — 현실적인 synthetic generator

### 4.1 Latent truth

- root concept 수는 우선 10으로 두되, 768차원 독립 랜덤 단위벡터를 그대로
  쓰지 않는다.
- 3–5개의 global semantic factor에서 correlated root를 만든다. 관련 root의
  cosine은 대략 `0.2–0.6`, 무관 root는 `-0.1–0.15` 범위가 되도록 생성 후
  실제 cosine 분포를 함께 저장한다.
- 각 노트는 2–3개 root의 soft mixture로 만들며, true membership vector를
  보존한다. 경계 노트가 충분히 생기도록 mixture concentration을 명시한다.
- content embedding은 `latent semantic + note-specific component + Gaussian
  noise` 뒤 L2 normalize로 생성한다. content noise는 독립적인 실험 축이다.

### 4.2 Observed tag

태그는 latent truth와 분리된 **관측값**으로 생성한다. 각 노트에 다수의
heterogeneous tag를 만들고, 다음 corruption을 독립적으로 기록한다.

- 누락 tag
- 다른 root의 오태그
- 지나치게 일반적인 tag
- 과잉 tag

기존 기준 확률 `(15%, 10%, 20%, 25%)`을 `c = 1.0`으로 정의하고, 각 event
확률을 `min(1, c × 기준 확률)`으로 둔다. 이 방식이면 corruption 수준을
바꾸더라도 어떤 종류의 오류가 증가했는지 추적할 수 있다. 태그 개수와 각
태그의 신뢰도도 metadata에 남겨 sum+normalize로 사라지는 정보를 점검한다.

### 4.3 검증 테스트

새 generator에는 최소한 다음 테스트를 둔다.

- 같은 seed에서 embeddings, memberships, tags, corruption flags가 동일함
- root cosine 분포가 설정한 관련/무관 범위를 충족함
- true memberships는 행별 합이 1이고, boundary subset이 비어 있지 않음
- `c = 0`에는 corruption이 없고 `c`가 증가하면 각 오류 유형의 기대 빈도가 증가함
- shuffled-tag control은 content와 row alignment를 제외한 태그-진실 관계를 끊음

## 5. Phase 2 — 실험 행렬

처음에는 true root 수 `K=10`을 알고 있다고 가정해 geometry와 membership을
진단한다. 각 조건은 SFCM seed와 generator seed를 분리하고 최소 3개 seed로
반복한다.

| 축 | 초기 값 |
| --- | --- |
| content noise | `0.05, 0.10, 0.20, 0.30, 0.40` |
| tag corruption multiplier `c` | `0, 0.5, 1.0, 1.5, 2.0` |
| tag weight | `0, 0.25, 0.5, 1.0, 2.0` |
| seeds | `42, 43, 44` |

`tag weight = 0`은 content-only 기준선으로 중복 실행하지 않고, 각
`(content noise, seed)`에 하나의 기준선으로 공유한다. 전체 3축 전수실험 전에는
중앙값 조건으로 smoke run을 수행해 generator/metric/fusion 구현을 검증한다.

각 조건에는 다음 control을 포함한다.

- **content-only:** 태그를 사용하지 않음
- **correct-tag:** 해당 조건의 관측 태그
- **shuffled-tag:** 태그 행을 permutation하여 우연한 벡터 추가 효과를 측정
- **oracle-tag (선택):** corruption 없는 관측 태그로 태그 신호의 상한을 확인

## 6. Phase 3 — fusion × PCA ablation

"태그 사용" 전체가 아니라 결합 지점과 연산을 분리해 비교한다.

| ID | 경로 | 확인할 가설 |
| --- | --- | --- |
| A | `content → PCA → SFCM` | 기준선 |
| B-add | `content ⊕ aggregated-tag → PCA → SFCM` | early additive fusion의 geometry 비용 |
| B-concat | `[content \| weight·tag] → PCA → SFCM` | 현 helper의 early concatenation 비용 |
| C | `content → PCA`, `tag → 같은 PCA transform`, 이후 결합 → SFCM | PCA fit 변화와 fusion 효과의 분리 |
| D | `content → PCA → SFCM`, tags는 cluster prior/reranking에만 사용 | 채널 분리 설계의 기준 |

`⊕`, tag aggregation, 정규화 시점, 결합 뒤 재정규화 여부를 코드와 결과 파일에
명시한다. C는 content/tag가 같은 차원을 가진 synthetic 조건에서만 먼저
실행하고, 실제 데이터처럼 차원이 다르면 정렬 transform을 별도 실험으로 다룬다.

## 7. Phase 4 — 평가 지표

### 7.1 Hard-label 보조 지표

true membership의 argmax를 dominant root로 변환해 ARI/NMI를 계산한다. 이는
기존 결과와 비교하기 위한 보조 지표이며, soft mixture 품질의 주 지표가 아니다.

### 7.2 Soft-membership 주 지표

예측 cluster와 true root를 Hungarian matching으로 정렬한 뒤 다음을 계산한다.

- membership cosine similarity
- Jensen–Shannon divergence
- MAE와 MSE
- dominant-1 accuracy
- top-2 root recall

다음 두 population을 반드시 분리해 보고한다.

- 전체 노트
- `max(true_membership) < 0.6`인 boundary 노트

boundary 결과가 전체 평균보다 훨씬 나쁘면, 태그 융합이 혼합 주제의 불확실성을
해소하지 못하거나 오히려 과도하게 hard assignment를 만들고 있다는 신호다.

### 7.3 K를 모르는 조건

진단 단계가 끝난 뒤 동일한 유망 조건에 대해 K 선택을 활성화한다. true K=10
고정 결과와 선택된 K, hard/soft metric, 안정성을 나란히 보고한다. K 고정
실험의 우수성이 실제 시스템의 cluster selection 우수성으로 확대 해석되지 않게
하기 위함이다.

## 8. Phase 5 — 분석과 의사결정

각 metric에 대해 `fusion score - content-only score`의 평균과 seed 간 신뢰구간을
계산한다. 주 산출물은 content noise와 tag corruption을 축으로 하고 tag weight를
패널로 둔 phase diagram이다.

- 유리: soft 주 지표가 content-only보다 개선되고, shuffled-tag보다 확실히 우수
- 중립: 신뢰구간이 0을 포함하거나 개선이 hard metric에만 국한됨
- 불리: soft 주 지표 또는 boundary subset에서 일관되게 하락

PCA ablation에서 B가 나쁘고 C가 회복되면 PCA fit의 축 변화가 주요 원인이다.
B와 C 모두 나쁘면 early fusion 자체의 표현 기하가 문제일 가능성이 높다.
D가 content-only를 훼손하지 않고 metadata 품질을 높이면 제품 설계는 채널 분리
방향으로 진행한다.

## 9. 구현 순서와 산출물

1. `synthetic_tag_fusion.py`에 generator, corruption 모델, fusion variants,
   matching/soft metrics를 추가한다.
2. `test_synthetic_tag_fusion.py`에 Phase 1의 결정성·분포·control 테스트를 추가한다.
3. `benchmark_synthetic_tag_fusion.py`에 CLI, sweep scheduler, JSON/CSV 저장을 추가한다.
4. 작은 smoke matrix로 결과 형식과 metric 방향성을 검증한다.
5. 3-seed 전체 sweep 및 K-unknown follow-up을 실행한다.
6. `benchmarks/synthetic-tag-fusion-YYYY-MM-DD/`에 `README.md`, `report.json`,
   `runs.csv`, phase diagram 이미지, 조건별 raw artifact를 저장한다.

기존 `benchmark_main_optimization_review.py`의 deterministic case/run/report
패턴을 재사용하되, 성능 회귀 비교용 benchmark와 태그 융합 가설 검증용 benchmark는
분리한다.

## 10. 완료 조건

다음이 충족되면 이 계획의 실험 단계는 완료다.

- 모든 generator와 metric 테스트가 통과하고 seed 재현성이 확인됨
- content noise × corruption × weight 결과가 3개 이상 seed에 대해 저장됨
- content-only, correct-tag, shuffled-tag, fusion/PCA ablation이 같은 표에 있음
- 전체와 boundary subset의 hard/soft metric이 함께 보고됨
- true-K와 K-unknown 결과가 구분됨
- phase diagram으로 태그 융합의 이득/손해 경계를 설명할 수 있음
- 최종 결론이 태그 일반론이 아니라 **특정 데이터 조건과 fusion 방식**에 한정돼 있음
