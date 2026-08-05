# 계층형 PCA + 구면 FCM 클러스터링/시각화 JS 이식 기획서

## 1. 목적

현재 Python 구현의 핵심 파이프라인만 JavaScript/TypeScript로 이식한다.

이식 대상은 다음 하나다.

```text
임베딩
  -> L2 정규화
  -> PCA 64차원
  -> PCA 결과 L2 정규화
  -> 구면 FCM
  -> 실루엣 기반 k 선택
  -> 재귀 분할
  -> 최대 4계층 계층 구조
```

시각화는 클러스터링 결과를 이용해 별도로 수행한다.

```text
원본 임베딩
  -> L2 정규화
  -> PCA 32차원
  -> PCA 결과 L2 정규화
  -> UMAP 2차원
  -> 계층형 색상/라벨 렌더링
```

모든 기존 평면 클러스터링 파이프라인을 이식할 필요는 없다. PCA + FCM 계층형 파이프라인과 그 시각화만 구현한다.

---

## 2. 권장 기술 구조

TypeScript + Node.js를 권장한다. 브라우저에서 실행할 경우 동일한 순수 함수 모듈을 번들링한다.

```text
src/
  data/
    loadEmbeddings.ts
    loadAssignments.ts
  math/
    normalize.ts
    pca.ts
    distances.ts
    seededRandom.ts
  clustering/
    sphericalFcm.ts
    noiseDetection.ts
    kSelection.ts
    hierarchy.ts
  visualization/
    projectUmap.ts
    labels.ts
    colors.ts
    renderScatter.ts
  cli/
    cluster.ts
    visualize.ts
  types.ts
```

각 모듈은 파일 하나의 책임만 가지도록 한다. FCM, 재귀 트리, UMAP 렌더러를 하나의 거대한 파일에 넣지 않는다.

### 라이브러리 원칙

- PCA는 SVD 또는 안정적인 PCA 라이브러리를 사용한다.
- UMAP은 `n_neighbors`, `min_dist`, `metric`, `spread`, `seed`를 지원하는 구현을 사용한다.
- Python 설정의 `densmap=true`를 지원하지 않는 JS UMAP 라이브러리를 사용한다면 자동으로 무시하지 말고 설정에 `densmapSupported`를 표시하거나 명확한 경고를 출력한다.
- 거리 계산은 가능한 한 `Float32Array`와 벡터화 루프를 사용한다.
- 무작위 초기화에는 `Math.random()`을 사용하지 말고 시드가 고정된 PRNG를 사용한다.

---

## 3. 기준 실행 설정

현재 데이터셋 결과를 재현하기 위한 기준 프로파일이다.

```json
{
  "seed": 42,
  "maxDepth": 4,
  "minNodeSize": 60,
  "minChildSize": 20,
  "minClusters": 2,
  "maxClusters": 4,
  "kSelection": "silhouette",
  "minMembership": 0.20,
  "distanceZ": 3.5,
  "minSplitSilhouette": 0.05,
  "clusteringPcaComponents": 64,
  "fcm": {
    "m": 2.0,
    "maxIter": 200,
    "tol": 0.000001
  },
  "visualization": {
    "pcaComponents": 32,
    "nNeighbors": 15,
    "minDist": 0.02,
    "metric": "cosine",
    "spread": 0.85,
    "densmap": true
  }
}
```

주의: Python 함수의 라이브러리 기본값에는 `maxClusters=8`, `minMembership=0.40`도 존재한다. JS 제품의 기본 실행값은 최신 실제 실행에 사용한 위의 재현 프로파일(`maxClusters=4`, `minMembership=0.20`)로 맞춘다. 두 값을 CLI에서 변경할 수 있게 한다.

---

## 4. 입력 데이터

### 4.1 기본 JSON 형식

최상위 배열을 지원한다.

```json
[
  {
    "id": 0,
    "tag": "World",
    "embedding": [0.01, -0.02, 0.03]
  }
]
```

`id`가 없으면 배열 인덱스를 사용한다. `embedding`을 제외한 필드는 메타데이터로 보존한다.

### 4.2 체크포인트 형식

다음 형식도 지원한다.

```json
{
  "records": [
    {
      "id": 0,
      "class": "Company",
      "class_hierarchy": "...>...>...",
      "text": "...",
      "embedding": [0.01, -0.02, 0.03]
    }
  ]
}
```

### 4.3 입력 검증

- 레코드 수가 0이면 실패한다.
- 모든 임베딩의 차원이 같아야 한다.
- `NaN`, `Infinity`, 비수치 값을 거부한다.
- 임베딩 차원은 PCA 차원보다 커야 하며, 실제 PCA 차원은 `min(requested, sampleCount, embeddingDim)`으로 제한한다.
- ID는 중복되면 실패한다.

---

## 5. 공통 수학 규칙

### 5.1 L2 정규화

각 행 벡터에 대해 다음을 수행한다.

```text
norm(x) = sqrt(sum(x[d]^2))
x_normalized = x / max(norm(x), epsilon)
```

`epsilon=1e-12`를 사용한다. 모든 정규화 이후 벡터의 길이는 약 1이어야 한다.

구면 FCM에서는 코사인 유사도를 직접 계산하지 않는다. 입력과 중심을 모두 단위 구면에 두고 유클리드 거리를 사용한다.

```text
||x - c||² = 2 - 2 * cosine(x, c)
```

따라서 단위 벡터 간 유클리드 거리와 코사인 거리는 순위가 동일하다.

### 5.2 PCA

클러스터링과 시각화에서 PCA를 각각 한 번 수행한다.

```text
Xn = L2Normalize(X)
Z = PCA.fitTransform(Xn)
Z = L2Normalize(Z)
```

현재 CLI는 입력을 먼저 정규화한 뒤 PCA를 수행하므로 JS도 이 순서를 따른다. 재귀 단계마다 PCA를 다시 계산하지 않는다. 전체 데이터에 대해 PCA를 한 번 계산한 뒤, 재귀 단계에서는 해당 PCA 행렬의 부분 행만 사용한다.

PCA는 열 평균을 제거한 뒤 SVD 기반으로 계산한다. PCA 결과의 부호는 임의적이므로, 부호가 달라져도 거리와 클러스터 품질이 유지되면 정상으로 간주한다.

---

## 6. 구면 FCM 구현

### 6.1 초기화

입력 `X`를 다시 L2 정규화한다.

```text
U[i][j] = seededRandom()
U[i] = U[i] / sum(U[i])
```

여기서 `U`의 각 행 합은 1이다. 기본값은 다음과 같다.

```text
m = 2.0
maxIter = 200
tol = 1e-6
```

### 6.2 중심 업데이트

각 반복마다 다음을 수행한다.

```text
W[i][j] = U[i][j]^m
C[j] = sum_i(W[i][j] * X[i])
C[j] = L2Normalize(C[j])
```

중심 벡터의 길이가 `epsilon`보다 작으면 입력 샘플 하나를 시드가 고정된 무작위 방식으로 골라 중심으로 사용한 후 정규화한다.

### 6.3 거리와 소속도 업데이트

```text
d[i][j] = EuclideanDistance(X[i], C[j])
d[i][j] = max(d[i][j], epsilon)
power = 2 / (m - 1)
U[i][j] = 1 / sum_l((d[i][j] / d[i][l])^power)
```

반복 종료 조건은 다음과 같다.

```text
max(abs(U_new - U_old)) < tol
```

`maxIter`에 도달하면 마지막 소속도를 사용한다. 최종 하드 라벨은 각 행에서
소속도가 가장 큰 클러스터의 인덱스다.

반환 타입 예시는 다음과 같다.

```ts
type FcmResult = {
  labels: Int32Array;
  memberships: Float32Array; // [sampleCount, k]
  centers: Float32Array;      // [k, dimension]
  iterations: number;
};
```

---

## 7. 노이즈 판정

각 FCM 후보에 대해 하드 라벨을 만든 뒤 노이즈를 판정한다.

### 7.1 소속도 기준

최대 소속도가 낮으면서 동시에 1위와 2위 소속도 차이가 작으면
경계 후보로 판정한다.

```text
max_j(U[i][j]) < minMembership
AND
largestMembership - secondLargestMembership < maxMembershipGap
```

두 조건 중 하나만 만족하는 문서는 경계 후보가 아니다.

### 7.2 클러스터별 거리 이상치 기준

각 클러스터에 대해 해당 클러스터로 배정된 샘플의 중심 거리 배열을 만든다.

```text
median = median(distances)
MAD = median(abs(distances - median))
robustScale = 1.4826 * MAD
threshold = median + distanceZ * robustScale
```

소속도 경계 후보가 `distance > threshold`까지 동시에 만족하면
`noise`, 거리 임계값 이내이면 `boundary`, 경계 후보가 아니면 `core`로
분류한다. 샘플 수가 4개 미만이거나 `MAD <= epsilon`이면 거리 임계값을
무한대로 두므로 경계 후보는 `boundary`가 된다.

세 신호의 노드 내 백분위 순위를 기하평균한 `noise_score`도 계산한다.
클러스터링 완료 후 전체 문서에서 점수가 높은 상위 1%를 추가 노이즈로
선정한다. 이 규칙은 클러스터 구조 학습에는 영향을 주지 않으며, 동점은 문서
ID 오름차순으로 결정한다. 임계값 기반 노이즈와 순위 기반 노이즈는 각각
`is_natural_noise`, `is_forced_noise`로 구분한다.

노이즈가 된 샘플은 이후 하위 재귀 클러스터링에 전달하지 않는다.

---

## 8. 가변 클러스터 수 `k` 선택

각 계층 노드에서 독립적으로 `k`를 선택한다.

### 8.1 후보 범위

```text
maximumK = min(maxClusters, floor(nodeSize / minChildSize))
k = minClusters ... maximumK
```

후보마다 다음 순서로 수행한다.

1. 구면 FCM 실행
2. 소속도 및 거리 기반 노이즈 판정
3. 노이즈가 아닌 샘플만 사용해 생존 클러스터를 계산
4. `생존 클러스터 수 >= 2`인 경우 실루엣 계수 계산

작은 클러스터는 `minChildSize`보다 작으면 생존 클러스터로 인정하지 않는다. 작은 클러스터의 샘플은 해당 후보에서 노이즈처럼 처리한다.

### 8.2 실루엣 선택

현재 기본 선택 방식은 실루엣 계수다.

```text
best = silhouette가 가장 큰 후보
동률이면 Xie-Beni 값이 작은 후보
그래도 동률이면 k가 작은 후보
```

실루엣은 노이즈 샘플을 제외한 정규화 PCA 특성에 대해 유클리드 거리로 계산한다.

### 8.3 무릎점 선택

`kSelection="knee"`일 때는 후보의 FCM 목적 함수 곡선을 사용한다.

1. 목적 함수를 후보 순서에 따라 `[0, 1]` 범위로 정규화한다.
2. 첫 후보와 마지막 후보를 잇는 직선을 만든다.
3. 각 후보와 직선 사이의 수직 거리를 계산한다.
4. 거리가 가장 큰 후보를 무릎점으로 선택한다.

실루엣이 유효한 후보만 무릎점 후보로 사용한다.

---

## 9. 계층 재귀

### 9.1 재귀 절차

루트 노드부터 시작한다.

```text
recurse(indices, node, depth):
  if depth >= maxDepth: stop(max_depth_reached)
  if indices.length < minNodeSize: stop(node_too_small)

  candidates = evaluateK(indices)
  if no valid candidate: stop(no_valid_silhouette_split)
  if best.silhouette < minSplitSilhouette:
    stop(silhouette_below_threshold)

  각 생존 클러스터에 대해:
    현재 계층에 0-based 클러스터 ID 기록
    자식 노드 생성
    자식에 대해 recurse()
```

노드에서 생존한 클러스터만 자식 노드가 된다. 노이즈는 자식 노드가 아니다.

### 9.2 계층 번호 규칙

내부 ID는 0부터 시작한다.

```text
level_1_cluster = 0, 1, 2, ...
level_2_cluster = 0, 1, 2, ...
```

시각화 라벨만 사람이 읽기 쉽게 1부터 시작한다.

```text
내부 경로: 2/1/3
표시 라벨: 3-2-4
```

노드에서 중간에 노이즈가 되면 다음과 같이 표시한다.

```text
내부 경로: 2/1/noise
표시 라벨: 3-2-noise
```

---

## 10. 구현 시 주의할 점

1. FCM은 매 반복마다 입력과 중심을 정규화해야 한다. 초기 한 번만 정규화하면 안 된다.
2. 코사인 거리를 별도 알고리즘으로 바꾸지 않는다. 단위 벡터에서 유클리드 거리를 사용한다.
3. PCA64와 시각화 PCA32를 혼동하지 않는다.
4. `cluster` 값만으로 계층 클러스터를 식별하지 않는다. 전역 식별자는 `cluster_path`다.
5. 내부 클러스터 번호는 0-based, 표시 라벨은 1-based다.
6. 노이즈를 하나의 일반 클러스터로 취급하지 않는다.
7. UMAP 좌표의 절대적인 방향/축 크기는 의미가 없다. 시드와 파라미터가 같은 경우의 상대적 구조만 비교한다.
8. 시각화 색상은 클러스터링 결과를 바꾸지 않는다.
9. 모든 평면 파이프라인을 JS로 이식하지 않는다. 이 문서의 범위는 계층형 PCA + 구면 FCM이다.

---

## 11. 할당 결과 출력 형식

CSV 또는 JSON으로 다음 필드를 출력한다.

```text
id
원본 메타데이터 필드들
level_1_cluster
level_2_cluster
level_3_cluster
level_4_cluster
cluster
cluster_path
is_noise
is_natural_noise
is_forced_noise
is_boundary
document_type
noise_score
boundary_level
noise_level
leaf_level
```

예시:

```csv
id,tag,level_1_cluster,level_2_cluster,level_3_cluster,level_4_cluster,cluster,cluster_path,is_noise,noise_level,leaf_level
0,World,0,0,0,0,0,0/0/0/0,false,-1,4
1,World,0,0,-1,-1,-1,0/0/noise,true,3,2
```

필드 의미:

- `level_N_cluster`: 해당 계층의 0-based ID. 해당 계층까지 도달하지 않았거나 노이즈면 `-1`.
- `cluster`: 마지막으로 도달한 계층의 ID. 노이즈면 `-1`. 전역적으로 유일한 ID가 아니므로 전역 식별에는 `cluster_path`를 사용한다.
- `cluster_path`: 0-based `/` 경로. 전역적으로 유일한 클러스터 경로다.
- `is_noise`: 불리언 값.
- `noise_level`: 노이즈가 된 계층 번호. 일반 샘플은 `-1`.
- `leaf_level`: 마지막으로 유효한 계층 번호. 도달하지 못한 샘플은 `-1`.

ID 기준으로 임베딩과 할당 결과를 일대일로 결합한다. 행 순서가 같다는 가정만으로 합치지 않는다.

---

## 12. 트리 JSON 출력 형식

최상위 구조:

```json
{
  "config": {},
  "summary": {},
  "root": {}
}
```

각 노드는 다음 필드를 가진다.

```json
{
  "node_id": "2/1",
  "parent_id": "2",
  "path": "2/1",
  "depth": 2,
  "size": 300,
  "selected_k": 2,
  "selected_silhouette": 0.094,
  "selected_valid_clusters": 2,
  "noise_count": 4,
  "candidate_metrics": [
    {
      "k": 2,
      "silhouette": 0.094,
      "xie_beni": 0.12,
      "objective": 0.45,
      "valid_clusters": 2,
      "noise_count": 4,
      "cluster_sizes": [110, 108]
    }
  ],
  "stop_reason": null,
  "children": []
}
```

가능한 `stop_reason`:

```text
max_depth_reached
node_too_small
too_few_samples_for_two_valid_children
no_valid_silhouette_split
silhouette_below_threshold
```

`summary`에는 최소한 다음을 포함한다.

```text
method
samples
pca_components
levels_requested
levels_reached
node_count
leaf_count
noise_count
noise_by_level
leaf_cluster_count
runtime_sec
```

---

## 13. 시각화 구현

### 13.1 전처리

클러스터링에서 사용한 PCA 결과를 재사용하지 않는다. 시각화는 다음의 별도 PCA32를 사용한다.

```text
Xn = L2Normalize(originalEmbeddings)
Z32 = PCA32.fitTransform(Xn)
Z32 = L2Normalize(Z32)
coordinates = UMAP2D.fitTransform(Z32)
```

기본 UMAP 설정:

```json
{
  "nNeighbors": 15,
  "minDist": 0.02,
  "metric": "cosine",
  "spread": 0.85,
  "densmap": true,
  "seed": 42
}
```

`pcaComponents`는 CLI에서 변경할 수 있지만 기본값은 32다.

### 13.2 비교 사전 설정

비교 화면에는 다음 네 가지를 표시한다.

| 이름 | n_neighbors | min_dist | spread | densmap |
|---|---:|---:|---:|---:|
| dense | 8 | 0.00 | 0.70 | true |
| compact | 12 | 0.01 | 0.80 | true |
| balanced | 15 | 0.02 | 0.85 | true |
| local | 20 | 0.03 | 0.90 | false |

### 13.3 색상 규칙

- 최상위 클러스터마다 서로 다른 색조를 사용한다.
- 같은 최상위 클러스터의 하위 클러스터는 동일한 색조의 밝기 차이로 표시한다.
- 하위 라벨의 정렬 순서는 문자열 정렬이 아니라 숫자 자연 정렬을 사용한다.
- 노이즈와 `*-noise`는 모두 회색 `#9aa0a6`로 표시한다.
- 최상위 클러스터 수가 10개 이하이면 구분 가능한 범주형 색상 팔레트를 사용한다.
- 라벨은 1-based 계층 경로로 표시한다.

예:

```text
cluster 3
cluster 3-1-1
cluster 3-1-2
cluster 3-2-1
cluster 3-2-noise
```

### 13.4 렌더링 요구사항

- 모든 입력 샘플은 정확히 하나의 좌표를 가진다.
- 할당 결과와 좌표를 ID 기준으로 결합한다.
- 범례에는 `cluster label (count)` 형식으로 각 라벨의 샘플 수를 표시한다.
- 축 이름은 `UMAP-1`, `UMAP-2`로 표시한다.
- 제목에는 `PCA-32 + UMAP`, 색상 기준, 계층형 여부를 표시한다.
- 브라우저에서는 마우스를 올렸을 때 `id`, 원본 메타데이터, 표시 라벨, `cluster_path`를 보여준다.
- Node 실행에서는 SVG 또는 HTML을 기본 출력으로 하고, PNG 출력은 선택 기능으로 둔다.

---

## 14. CLI 설계

클러스터링과 시각화를 분리한다.

```bash
# 계층형 PCA + FCM
node dist/cli.js cluster \
  --input data.json \
  --output-dir results \
  --max-depth 4 \
  --max-clusters 4 \
  --min-membership 0.20 \
  --pca-components 64 \
  --seed 42

# 단일 시각화
node dist/cli.js visualize \
  --input data.json \
  --assignments results/hierarchical_assignments.csv \
  --output results/scatter.html \
  --pca-components 32 \
  --n-neighbors 15 \
  --min-dist 0.02 \
  --spread 0.85 \
  --densmap

# 네 가지 UMAP 비교
node dist/cli.js visualize \
  --input data.json \
  --assignments results/hierarchical_assignments.csv \
  --comparison \
  --output results/umap_comparison.html
```

클러스터링 CLI는 시각화를 자동 실행하지 않는다. 두 작업을 분리해야 대형 데이터셋에서 재클러스터링 없이 시각화 설정만 바꿀 수 있다.

---

## 15. 성능 요구사항

- PCA64는 전체 데이터에 대해 한 번만 수행한다.
- 재귀 호출마다 PCA를 다시 계산하지 않는다.
- FCM의 중심 거리 계산은 `O(nodeSize * k * dimension)`이다.
- 실루엣 계산은 기본적으로 정확한 방식으로 구현하되, 대형 데이터셋을 위해 `silhouetteSampleSize` 또는 블록 단위 계산 옵션을 제공한다.
- UMAP 비교 모드는 의도적으로 UMAP을 네 번 실행한다. 일반 모드는 한 번만 실행한다.
- 메모리상 원본 임베딩, PCA64 행렬, PCA32 행렬을 동시에 복제하지 않도록 형식화 배열과 뷰를 사용한다.
- 긴 작업은 진행률을 출력한다.

---

## 16. 테스트 및 검증

### 단위 테스트

- L2 정규화 후 각 행의 노름이 1에 가깝다.
- FCM 소속도 각 행의 합이 1에 가깝다.
- FCM 중심의 노름이 1에 가깝다.
- 동일한 시드로 실행하면 동일한 결과가 나온다.
- 모든 생존 자식 클러스터의 크기가 `minChildSize` 이상이다.
- 노이즈 샘플은 하위 재귀에 다시 들어가지 않는다.
- `level_N_cluster=-1` 이후의 계층 값도 모두 `-1`이다.
- 트리의 자식 경로는 부모 경로의 접두사다.
- 할당 결과의 `cluster_path`와 트리의 리프 경로가 일치한다.
- ID 순서가 섞여 있어도 행 순서가 아니라 ID 기준으로 시각화 데이터가 매칭된다.

### 기준 데이터 회귀 테스트

AG News 4,000개 데이터셋을 기준으로 다음을 확인한다.

```text
samples: 4000
levels_reached: 4
leaf_cluster_count: 25
noise_count: 110
```

현재 Python 기준 실행의 참고 지표:

```text
flat PCA64 + FCM
NMI: 0.5669
ARI: 0.6056
silhouette: 0.0811
```

JS의 PRNG, PCA SVD, UMAP 구현이 Python과 다르면 샘플별 라벨 번호가 달라질 수 있다. 이 경우 원시 라벨 배열을 그대로 비교하지 말고 다음을 비교한다.

- 계층 깊이
- 리프 수
- 노이즈 비율
- 실루엣/NMI/ARI 범위
- 같은 최상위 주제의 분리 구조
- 시각화 좌표의 NaN 여부와 전체 샘플 수

---

## 17. 완료 기준

- [ ] JSON 배열과 `records` 체크포인트를 모두 읽는다.
- [ ] PCA64 + 구면 FCM + 재귀 계층화를 실행한다.
- [ ] 실루엣/무릎점 기반 가변 `k`를 지원한다.
- [ ] 소속도/거리/최소 크기 기반 노이즈 처리를 지원한다.
- [ ] 최대 4계층과 조기 중단을 지원한다.
- [ ] 할당 결과 CSV/JSON과 트리 JSON을 출력한다.
- [ ] `3-2-4` 형식의 표시 라벨을 생성한다.
- [ ] 최상위 색조 분리와 하위 색조 계열 색상을 구현한다.
- [ ] PCA32 + UMAP 시각화를 구현한다.
- [ ] balanced 기본 UMAP 설정과 네 가지 비교 사전 설정을 지원한다.
- [ ] 기준 데이터셋 회귀 테스트를 통과한다.
- [ ] 클러스터링 CLI와 시각화 CLI를 분리한다.
