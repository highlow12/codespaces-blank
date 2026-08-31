# Atomic Clusters 멀티모달 임베딩 계획

기준일: 2026-08-30

이 문서는 Atomic Clusters에 이미지 임베딩과 멀티모달 클러스터링을 추가하기 위한 제품/구현 계획이다.

핵심 원칙은 **서로 다른 임베딩 모델의 공간을 정렬하거나 합치지 않는 것**이다. 하나의 Vault clustering state는 언제나 정확히 하나의 embedding model identity에만 속한다. 텍스트와 이미지를 함께 다루려면 해당 모델 자체가 두 modality를 같은 embedding space에 투영해야 한다.

---

# 1. 핵심 결정

## 1.1 하나의 활성 임베딩 공간만 사용

한 번의 clustering state에는 다음이 성립해야 한다.

```text
모든 vector
  → 같은 provider
  → 같은 model
  → 같은 revision/version
  → 같은 embedding dimension
  → 같은 preprocessing contract
```

따라서 다음 방식은 지원하지 않는다.

```text
text model A embedding
+ image model B embedding
+ learned linear transform
→ shared space
```

또한 zero padding, 임의 projection, Procrustes alignment, anchor 기반 선형 정렬 등 서로 다른 모델 공간을 합치는 기능도 제품 범위에서 제외한다.

이 결정의 이유는 다음과 같다.

- 서로 다른 임베딩 공간의 정렬 품질을 별도로 검증할 필요가 없다.
- model 간 semantic scale 차이가 clustering 결과를 오염시키지 않는다.
- PCA/HDBSCAN 결과의 의미를 설명하기 쉬워진다.
- cache invalidation 규칙이 단순해진다.
- 멀티모달 품질 문제를 embedding model 자체의 품질로 한정할 수 있다.

---

# 2. 멀티모달 입력 범위

## 2.1 1차 지원 modality

첫 구현은 다음 두 종류만 지원한다.

- Markdown/Text
- Image

이미지 확장자 baseline:

```text
.png
.jpg
.jpeg
.webp
```

GIF, SVG, HEIC, RAW 등은 모델/runtime 호환성이 검증된 뒤 추가한다.

영상과 음성은 현재 단계에서 지원하지 않는다. 이미지 지원이 실제 Vault에서 충분히 안정된 뒤 별도 modality로 확장한다.

## 2.2 지원 조건

사용자가 선택한 embedding model이 text와 image를 같은 semantic embedding space에 넣을 수 있을 때만 두 modality를 함께 clustering한다.

예:

```text
Text note ─┐
           ├→ Multimodal Embedding Model → shared vectors → PCA → UMAP → HDBSCAN
Image ─────┘
```

반대로 선택한 모델이 text-only이면 이미지 파일은 clustering 대상에 넣지 않는다.

UI에서 명확히 표시한다.

```text
Embedding model: multilingual-e5-small
Supported: Text
Ignored: Images
```

멀티모달 모델이면:

```text
Embedding model: <multimodal model>
Supported: Text, Image
```

---

# 3. Embedding Model Identity

현재 provider/model 단위 cache key를 더 엄격한 `EmbeddingSpaceIdentity`로 확장한다.

예시:

```ts
interface EmbeddingSpaceIdentity {
  provider: string;
  model: string;
  revision: string;
  dimension: number;
  modalities: Array<"text" | "image">;
  preprocessingVersion: string;
}
```

실제로 동일 공간인지 판정할 때 표시 이름만 비교하지 않는다.

최소 다음 값이 같아야 compatible하다고 본다.

```text
provider
model
revision
embedding dimension
preprocessing version
```

이 identity 전체의 stable hash를 생성한다.

```text
embedding_space_id = hash(EmbeddingSpaceIdentity)
```

모든 embedding과 clustering result에 `embedding_space_id`를 저장한다.

---

# 4. 모델 변경 정책 — 전체 재임베딩 + 전체 재클러스터링

## 4.1 절대 규칙

**embedding_space_id가 달라지면 기존 vector를 새 run에서 단 하나도 재사용하지 않는다.**

즉 다음 변경은 모두 full invalidation이다.

- provider 변경
- model 변경
- model revision 변경
- embedding dimension 변경
- image/text preprocessing contract 변경
- tokenizer/image processor의 의미 있는 version 변경

실행 흐름:

```text
Embedding model 변경
        ↓
기존 embedding space와 불일치 감지
        ↓
새 모델로 전체 active item 재임베딩
        ↓
새 PCA fit
        ↓
새 UMAP fit
        ↓
새 HDBSCAN
        ↓
새 hierarchy
        ↓
새 cluster titles
        ↓
새 visualization
```

기존 model의 cache를 반드시 즉시 삭제할 필요는 없다. provider/model별 cache namespace로 남겨 두었다가 사용자가 이전 모델로 돌아오면 재사용할 수 있다.

하지만 **한 clustering result 안에서 두 cache namespace의 vector를 섞는 것은 금지한다.**

## 4.2 모델 변경 UX

Settings에서 모델을 변경하면 즉시 다음 경고를 보여 준다.

```text
Changing the embedding model changes the semantic vector space.
All active notes and supported media must be re-embedded and clusters rebuilt.

Estimated items: 4,218
Text: 3,741
Images: 477
```

선택지:

- `Change and rebuild`
- `Change model only` — 설정만 변경하고 현재 Explorer는 stale 상태로 유지
- `Cancel`

`Change model only` 상태에서는 기존 결과를 새 모델 결과인 것처럼 보여 주지 않는다.

Explorer 상단:

```text
Clusters are based on the previous embedding model.
Rebuild required.
```

---

# 5. 공통 Item 모델

기존 `NoteRecord` 중심 구조를 장기적으로 일반화한다.

예시:

```ts
type ClusterItemKind = "markdown" | "image";

interface ClusterItemRecord {
  id: string;
  path: string;
  kind: ClusterItemKind;
  title: string;
  contentHash: string;
  mtime: number;
  searchableText?: string;
  metadata?: Record<string, unknown>;
}
```

Markdown은 기존 note content를 사용한다.

Image는 binary 자체를 `contentHash` 계산에 사용하고 embedding provider에 binary/image tensor를 전달한다.

`id`는 path와 분리하는 방향을 우선한다. rename만으로 semantic identity가 완전히 사라지지 않게 하기 위함이다.

---

# 6. 이미지 수집

## 6.1 Vault scanner

기존 Markdown 수집 단계에 media scanner를 추가한다.

```text
Vault
├─ *.md
├─ *.png
├─ *.jpg
├─ *.jpeg
└─ *.webp
```

excluded folder 규칙은 Markdown과 이미지에 동일하게 적용한다.

추가 설정:

```text
Include images in clustering: On / Off
Maximum image size
Supported extensions
```

## 6.2 이미지 hash

이미지는 binary content hash를 사용한다.

mtime만으로 cache validity를 판단하지 않는다.

```text
same binary hash
+ same embedding_space_id
→ embedding cache reuse
```

이미지 rename만 발생하고 binary hash가 같다면 embedding을 다시 계산하지 않는다.

---

# 7. 이미지 전처리

전처리는 embedding model contract에 포함한다.

예:

- resize
- crop/pad
- color conversion
- normalization

중요한 전처리 변경은 `preprocessingVersion`을 변경하여 기존 embedding을 invalidation한다.

제품 코드에서 모델이 요구하는 preprocessing을 임의로 통일하지 않는다. provider adapter가 자신의 model contract를 책임진다.

---

# 8. Embedding Provider 인터페이스 확장

현재 text embedding provider를 modality-aware provider로 일반화한다.

개념 예시:

```ts
interface EmbeddingProvider {
  id: string;
  model: string;
  space: EmbeddingSpaceIdentity;

  supports(kind: ClusterItemKind): boolean;

  embedText?(items: TextEmbeddingInput[]): Promise<CachedEmbedding[]>;
  embedImages?(items: ImageEmbeddingInput[]): Promise<CachedEmbedding[]>;
}
```

또는 내부적으로 공통 `embed(items)` API를 사용하되 item kind별 adapter를 둔다.

핵심 invariant:

```text
provider가 반환하는 모든 vector는
provider.space가 정의한 동일한 semantic space에 속해야 한다.
```

runtime에서 dimension mismatch가 하나라도 발견되면 clustering을 중단한다.

---

# 9. Cache 구조

cache key는 다음처럼 확장한다.

```text
embedding_space_id:item_id:content_hash
```

또는 DB column으로 분리한다.

```text
embeddings
- embedding_space_id
- item_id
- content_hash
- modality
- dimension
- vector
- updated_at
```

다른 모델의 embedding cache는 동시에 DB에 존재할 수 있다.

예:

```text
space_A / note1
space_A / image1
space_B / note1
space_B / image1
```

하지만 active clustering result는 하나의 `embedding_space_id`만 참조한다.

DB invariant test를 추가한다.

```text
COUNT(DISTINCT embedding_space_id) for one cluster result == 1
```

---

# 10. 부분 업데이트 정책과의 결합

기존 `plan.md`의 automatic refresh 정책과 다음 규칙으로 결합한다.

## 같은 embedding model일 때

```text
새 Markdown
→ 새 note만 embedding
→ provisional placement 또는 rebuild policy

수정 Markdown
→ 해당 note만 re-embedding

새 Image
→ 해당 image만 embedding

수정 Image
→ 해당 image만 re-embedding

rename + 동일 content hash
→ embedding reuse
```

## embedding model이 바뀌었을 때

부분 업데이트를 사용하지 않는다.

```text
model changed
→ incremental refresh 금지
→ provisional placement 금지
→ PCA reuse 금지
→ visualization reuse 금지
→ 전체 재임베딩
→ 전체 재클러스터링
```

이 규칙은 heuristic이 아니라 hard invariant로 구현한다.

---

# 11. 이미지의 cluster title 처리

현재 keyword cluster title generator는 Markdown text를 주 신호로 사용한다.

이미지만 포함된 cluster에서는 본문 keyword가 없으므로 별도 fallback이 필요하다.

1차 정책:

1. cluster에 Markdown이 있으면 기존 contrastive keyword title 사용
2. Markdown이 없거나 text signal이 부족하면 image filename과 Obsidian metadata 사용
3. 그래도 신호가 없으면 대표 이미지 filename을 fallback title로 사용

예:

```text
IMG_3812.jpg
IMG_3815.jpg
cat_sleeping.png
```

이라면 가능한 범위에서 filename token을 사용한다.

이미지 자체를 vision LLM으로 captioning하여 title을 만드는 기능은 필수 기능으로 넣지 않는다. embedding model이 이미지를 이해하는 것과 생성형 모델을 호출하는 것은 별개의 기능으로 유지한다.

향후 필요성이 확인되면 opt-in image caption metadata를 별도 계층으로 추가할 수 있다.

---

# 12. Explorer UI

## 12.1 Item type 표시

Explorer에서 Markdown과 Image를 시각적으로 구별한다.

예:

- Markdown: note icon
- Image: image icon 또는 thumbnail

cluster layout 계산 자체는 modality에 따라 달라지지 않는다.

## 12.2 이미지 선택

이미지를 선택하면 detail panel에 다음을 표시한다.

```text
thumbnail
filename/path
image dimensions
cluster
probability/outlier
related items
Open image
```

관련 item에는 Markdown과 다른 이미지가 모두 나올 수 있다.

즉 멀티모달 기능의 가장 중요한 사용자 경험 중 하나는 다음이다.

```text
이미지 선택
→ 의미적으로 관련된 Markdown 노트 발견
```

또는 반대 방향:

```text
Markdown 노트 선택
→ 같은 주제를 표현하는 이미지 발견
```

---

# 13. 검색과 필터 확장

기존 검색에 modality qualifier를 추가한다.

```text
type:note
type:image
```

필터 chip:

- All
- Notes
- Images

이미지는 기본적으로 filename/path/metadata lexical search를 지원한다.

semantic image search는 현재 clustering 결과를 이용한 related-items 탐색으로 먼저 제공하고, 별도의 vector query UI는 후순위로 둔다.

---

# 14. 사용자 수정 기능과 멀티모달

기존 manual correction 기능은 이미지에도 동일하게 적용한다.

- image exclude
- image preferred cluster
- cluster manual title
- manual group

사용자가 이미지를 제외한다고 원본 파일을 이동하거나 삭제하지 않는다.

---

# 15. 모델 변경과 사용자 override

embedding model 변경 후 cluster 구조는 완전히 다시 만들어지므로 node id와 cluster fingerprint가 크게 바뀔 수 있다.

manual cluster title/group을 무조건 새 구조에 이식하면 잘못된 의미를 붙일 위험이 있다.

따라서 모델 변경은 일반 full rebuild보다 더 보수적으로 처리한다.

정책:

- note/image 단위 exclusion은 그대로 보존
- item identity 기반 사용자 metadata는 보존
- cluster-level manual title/group은 자동으로 강제 승계하지 않음
- 이전/새 cluster member overlap이 충분히 높을 때만 승계 후보로 제시

예:

```text
Previous manual title: "Unity Rendering"
Possible matching new cluster: 82% member overlap
[Apply] [Ignore]
```

자동 적용 baseline은 모델 변경에서는 사용하지 않는 것을 기본값으로 한다.

---

# 16. 실패와 atomic rebuild

모델 변경으로 전체 재임베딩할 때 기존 결과를 먼저 삭제하지 않는다.

```text
Current result (model A) — active
        ↓
model B embedding staging
        ↓
PCA / clustering staging
        ↓
validation
        ↓
successful transaction
        ↓
model B result becomes active
```

중간에 다음 문제가 생겨도:

- 모델 다운로드 실패
- 이미지 decode 실패
- 일부 embedding 실패
- 메모리 부족
- 사용자 cancel
- clustering 실패

기존 model A Explorer는 그대로 사용할 수 있어야 한다.

단, UI에는 설정 모델과 현재 active result model이 다름을 보여 준다.

---

# 17. 전체 재임베딩 진행 UX

멀티모달 모델 변경은 가장 비싼 작업 중 하나이므로 진행 상황을 분리해서 보여 준다.

예:

```text
Rebuilding embedding space

Text      2,104 / 3,741
Images      183 /   477
Embedding   51%
PCA          pending
Clustering   pending
Titles       pending
```

Gemini/외부 API 계열 provider라면 예상 전송 item 수와 modality를 rebuild 시작 전에 보여 준다.

로컬 모델이면 모델 설치 여부와 예상 memory requirement를 preflight한다.

---

# 18. 성능 정책

이미지는 text보다 decode/preprocessing 비용과 memory 사용량이 클 수 있다.

측정해야 할 benchmark:

```text
1,000 text
1,000 image
500 text + 500 image
3,000 text + 1,000 image
10,000 mixed items
```

각 단계에서 측정:

- image read/decode
- preprocessing
- embedding throughput
- cache size
- PCA
- UMAP
- HDBSCAN
- peak RSS
- cancellation latency

thumbnail 생성은 embedding pipeline과 분리하여 Explorer에서 lazy load한다.

---

# 19. 검증

## 공간 일관성

- 서로 다른 `embedding_space_id`가 한 clustering input에 섞이면 실패
- dimension mismatch 즉시 실패
- model revision 변경 시 full invalidation
- preprocessing version 변경 시 full invalidation

## cache

- 동일 이미지 hash + 동일 model → cache hit
- rename + 동일 binary → cache hit
- 이미지 binary 변경 → cache miss
- model 변경 → active run에서는 이전 cache 사용 금지
- 이전 model로 되돌리면 해당 namespace cache 재사용 가능

## multimodal semantic smoke test

작은 fixture를 만든다.

```text
text: "고양이가 소파에서 자고 있다"
image: sleeping-cat.jpg

text: "Unity terrain rendering"
image: terrain-screenshot.png
```

같은 개념의 text/image pair가 임의의 unrelated pair보다 가까운지 검증한다.

이 테스트는 clustering 정답률 benchmark가 아니라 provider integration이 실제 shared space를 사용하고 있는지 확인하는 smoke test다.

---

# 20. 완료 기준

멀티모달 1차 기능은 다음 조건을 만족하면 완료로 본다.

1. 사용자가 text+image를 지원하는 하나의 embedding model을 선택할 수 있다.
2. Markdown과 이미지가 같은 embedding space에서 clustering된다.
3. 서로 다른 embedding model의 vector가 한 result에서 절대 섞이지 않는다.
4. embedding model identity가 바뀌면 전체 active item을 재임베딩한다.
5. 모델 변경 시 PCA/UMAP/HDBSCAN/hierarchy/title/visualization을 전부 새로 계산한다.
6. 이전 clustering result는 새 rebuild가 성공할 때까지 보존된다.
7. 이미지 추가/수정은 같은 model 안에서는 changed-only embedding을 사용한다.
8. 이미지 rename은 binary가 같으면 embedding을 재사용한다.
9. Explorer에서 Markdown과 이미지를 함께 탐색할 수 있다.
10. image-only cluster에도 최소한의 fallback title이 존재한다.

---

# 21. 구현 순서

## Milestone MM-A — 데이터 모델 일반화

- `NoteRecord` 의존성을 `ClusterItemRecord` 방향으로 일반화
- item kind 추가
- `embedding_space_id` 도입
- DB migration
- clustering input 단일-space invariant

## Milestone MM-B — 이미지 수집과 cache

- image scanner
- binary hash
- image metadata
- excluded folder/note 규칙 통합
- cache tests

## Milestone MM-C — 멀티모달 provider

- modality-aware provider interface
- 선택한 multimodal model adapter
- text/image preprocessing
- dimension/space validation
- integration smoke test

## Milestone MM-D — 전체 모델 교체 rebuild

- model-change invalidation
- staged full re-embedding
- staged full clustering
- atomic result switch
- stale Explorer banner
- cancel/recovery

## Milestone MM-E — Explorer

- image node/icon/thumbnail
- image detail panel
- mixed related items
- type filters
- image fallback titles

## Milestone MM-F — 성능 검증

- mixed Vault benchmark
- cache/storage size
- image embedding throughput
- peak memory
- large Vault cancellation UX

---

# 22. 기존 제품화 계획과의 우선순위

기존 `plan.md`의 UI/UX, 검색, 사용자 교정, 자동 refresh가 현재 제품의 1차 제품화 작업이다.

멀티모달은 그 위에 추가되는 다음 기능 축으로 둔다.

권장 순서:

```text
Explorer 안정화
→ Search / Focus
→ Manual corrections
→ Automatic refresh
→ Multimodal item model
→ Multimodal embedding/image clustering
→ Large mixed-Vault hardening
```

다만 `embedding_space_id`와 **모델 변경 시 전체 재임베딩/재클러스터링 규칙**은 멀티모달 구현 전에 먼저 도입해도 좋다. 이는 현재 text-only provider를 바꿀 때도 잘못된 embedding space 혼합을 구조적으로 막아 주기 때문이다.
