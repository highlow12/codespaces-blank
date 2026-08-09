# Main 브랜치 최적화 종합 리뷰 (2026-08-09)

## 결론

`a10d936`(최적화 전)과 `0c4613a`(현재 main)를 동일한 2 CPU 환경에서 비교했다.
Gemini 3,000건 실데이터를 주 검증셋으로 쓰고, 한 데이터 분포에 대한 과최적화를
찾기 위해 불균형·동률·경계 노이즈·저랭크 중복·미소 군집 합성셋 5종을 추가했다.
총 64개 독립 프로세스 실행이 모두 성공했다.

현재 기본 JSON 경로의 wall p50은 1,000건에서 48.7%, 3,000건에서 53.5%
감소했다. 캐시 경로를 사용하면 최적화 전 JSON 경로 대비 각각 69.6%, 61.3%
감소했다. 상태 파일은 두 크기 모두 약 49% 작아졌다. 합성셋에서도 현재 버전의
wall time은 31.4~65.0% 감소했으므로, 성능 이득이 Gemini 데이터의 라벨이나 특정
분포에만 맞춰진 증거는 발견하지 못했다.

다만 결과가 완전히 동일한 것은 아니다. exact 경로와 Gemini 300건 fast 경로는
표본별 path가 100% 일치했지만, Gemini 1,000/3,000건 fast 경로와 합성셋 일부는
fast 후보 선택 변화 때문에 세부 계층이 달라졌다. 군집 분할 ARI는 Gemini에서
0.9491 이상이었고 합성셋에서 0.9557 이상이었다. 저랭크 중복과 미소 군집
합성셋의 truth ARI는 각각 0.0058, 0.0024 하락했다. 따라서 결론은 “과최적화
증거 없음”이지 “모든 분포에서 의미 보존을 증명함”은 아니다.

## 무엇을 최적화했는가

변경은 다음 네 부류다.

1. FCM 수학·배열 계산
   - membership 갱신의 cluster-pair 비율 텐서를 정규화된 역거듭제곱으로 바꿔
     메모리 복잡도를 줄였다.
   - 단위 구면 거리를 `2 - 2·dot`으로 계산하고 제곱거리의 불필요한 제곱근을
     제거했다.
   - 후보 선택, validity, 최종 분류 사이에서 membership·distance artifact를
     재사용했다.
2. 탐색량 축소
   - fast FCM이 silhouette proxy와 명확한 scout winner를 이용해 refine 후보를
     줄인다.
   - 안정적인 부모 fuzzifier `m`을 자식에서 재사용하고 불안정할 때만 다시 찾는다.
   - PCA 차원 탐색에서 각 `k`마다 수행하던 이웃 검색을 최대 `k` 한 번과 prefix
     view로 통합했다.
3. 입력·시작 비용
   - 시각화를 생략할 때 UMAP/plotting 의존성을 lazy import한다.
   - gzip JSON 전체 materialization 대신 row-addressable float32 캐시에서 필요한
     행만 읽을 수 있게 했다.
4. 메모리·상태 크기
   - 입력, PCA projection, state embedding은 기본 float32로 저장하되 FCM 중심,
     membership, 목적 함수와 XB 누산은 float64를 유지한다.
   - 사용하지 않는 conditional path membership은 기본적으로 생성하지 않고 opt-in
     출력으로 남겼다.

병렬 FCM과 300건 표본 기반 K 선택 전략도 실험했지만, 각각 현재 2 CPU 환경에서
느려지거나 전체 선택과의 ARI가 낮아 기본 경로에는 넣지 않았다.

## 측정 프로토콜

- baseline: `a10d936a0764b775ddd89d57357bac304c3e43f0`
- current: `0c4613ab9fad0b5fb07274f68e7c59e05e1b82eb`
- 실제 데이터: `dbpedia_gemini_embeddings.json.gz`, 3,000 × 3,072,
  SHA-256 `9a949bec1402b52f4b2cba4376ea3eda7c69003b33b7b1ea72e9501cf84d25fc`
- 환경: Python 3.12.1, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0,
  Linux x86_64, 2 CPU
- thread: OpenBLAS/OMP/MKL/NumExpr 모두 2로 고정
- 성능: Gemini 1,000/3,000건, seed와 sample seed 42, 독립 프로세스 5회 p50
- 품질: Gemini fast 300/1,000건 seed 42·43·44, exact 300건 seed 42·43
- 보수 설정: 현재 코드에 float64 저장, `m` 재사용 비활성화, conditional
  membership 활성화를 동시에 적용
- 캐시 생성 시간은 fit wall time에서 제외했다. JSON fallback은 별도로 같은 표본을
  매번 읽어 end-to-end 비용에 포함했다.

실행기는 [`benchmark_main_optimization_review.py`](../benchmark_main_optimization_review.py),
원시 결과는 [`report.json`](../benchmarks/main-optimization-review-2026-08-09/report.json)과
[`runs.csv`](../benchmarks/main-optimization-review-2026-08-09/runs.csv)에 있다.

## 성능 결과

| rows | 버전/입력 | wall p50 | baseline 대비 | peak RSS p50 | RSS 변화 | state p50 |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | baseline JSON | 30.196초 | 기준 | 857,308 KiB | 기준 | 39,237,580 B |
| 1,000 | current JSON | 15.501초 | -48.7%, 1.95× | 830,232 KiB | -3.2% | 20,013,076 B |
| 1,000 | current cache | 9.181초 | -69.6%, 3.29× | 317,748 KiB | -62.9% | 20,023,121 B |
| 3,000 | baseline JSON | 73.360초 | 기준 | 1,245,568 KiB | 기준 | 92,252,843 B |
| 3,000 | current JSON | 34.102초 | -53.5%, 2.15× | 984,568 KiB | -21.0% | 47,184,000 B |
| 3,000 | current cache | 28.422초 | -61.3%, 2.58× | 540,984 KiB | -56.6% | 47,214,009 B |

캐시는 현재 JSON 경로와 비교해 1,000건에서 wall 40.8%·RSS 61.7%, 3,000건에서
wall 16.7%·RSS 45.1%를 추가로 줄였다. 캐시와 JSON의 cluster path, noise,
PCA 차원과 root K는 모든 10개 성능 실행 쌍에서 동일했다.

현재 JSON과 보수 설정의 1,000건 3개 시드 p50은 각각 16.156초와 16.747초였다.
보수 설정은 conditional membership과 float64 embedding 때문에 state p50이
20,148,166 B에서 39,808,626 B로 커졌다. RSS는 JSON 전체 로딩이 지배해 거의
같았다. 이 묶음 ablation은 세 옵션의 개별 기여를 분리하지는 않는다.

## 결과 보존성과 품질

### Gemini

- exact 300건 2개 시드: path·noise·PCA·K가 모두 정확히 일치했다. 중심 최대 절대
  차이는 `9.04e-06` 이하였다.
- fast 300건 3개 시드: path·noise·PCA·K가 모두 정확히 일치했다.
- fast 1,000건 3개 시드: baseline/current 분할 ARI는
  `0.9491~0.9784`, NMI는 `0.9775~0.9886`이었다. noise 차이는 없었다.
- fast 3,000건: path 문자열의 정확 일치율은 0.2693이지만 분할 ARI 0.9828,
  NMI 0.9810이었다. 이는 계층 경로 번호와 일부 하위 분기가 바뀐 영향이 크므로
  정확 path 비율만 품질 지표로 해석하면 안 된다.
- 1,000건 3개 시드의 외부 tag/class 품질 평균 변화는 tag ARI -0.0019,
  tag NMI -0.0009, class ARI -0.0028, class NMI +0.0017이었다. 시드별로 양·음
  변화가 모두 있어 일관된 품질 향상이나 붕괴는 관찰되지 않았다.
- 현재 기본 설정과 보수 설정은 3개 시드에서 ARI 0.9937 이상이었다. 두 시드는
  의미상 같은 분할(ARI 1.0)이었고 한 시드에서 3개 표본의 path가 달랐다.

### 어려운 합성 데이터

합성 embedding은 모두 행 단위 정규화되고 생성 seed가 고정된다.

| 케이스 | 스트레스 조건 | baseline→current wall | truth ARI baseline→current | 분할 ARI |
|---|---|---:|---:|---:|
| imbalanced overlap | 52:20:7:1 불균형, 근접 중심 | 5.906→3.717초 (-37.1%) | 0.9956→0.9956 | 1.0000 |
| duplicate ties | 정확 중복, 등거리 bridge | 4.976→3.412초 (-31.4%) | 0.9502→0.9502 | 1.0000 |
| boundary and noise | 겹치는 중심, 경계·균일 noise | 11.766→4.749초 (-59.6%) | 0.3926→0.3926 | 1.0000 |
| rank deficient duplicates | 관측 96D·내재 rank 5, 20% 중복 | 13.382→4.685초 (-65.0%) | 0.1473→0.1416 | 0.9557 |
| tiny clusters high K | 8개 군집, 최소 5건 | 7.294→4.318초 (-40.8%) | 0.9918→0.9894 | 0.9933 |

보수 설정과 현재 기본 설정은 합성 5종 모두 path와 분할이 정확히 일치했다.
따라서 이 합성셋에서 float32 저장, `m` 재사용, conditional membership 생략이
관찰된 baseline 차이의 원인은 아니다. baseline과 current의 fast 후보 평가 및
refine 정책 자체가 달라진 영향으로 보는 것이 타당하다.

`duplicate_ties`에서는 root K가 6에서 3으로 바뀌었지만 최종 분할 ARI가 1.0이고
truth 품질도 같았다. 반면 저랭크·미소 군집에서는 작은 품질 하락이 있으므로 fast
선택 경로를 strict semantic-preserving 최적화로 분류하면 안 된다.

## 과최적화 판정과 남은 위험

판정은 다음과 같다.

- 계산·입력·저장 최적화가 합성 5종에서도 모두 빨랐고 exact 결과가 보존됐다.
  특정 Gemini 라벨에 직접 의존하는 코드도 없어 데이터셋 전용 과최적화 증거는 없다.
- fast 선택 변화는 성능 최적화인 동시에 모델 선택 정책 변화다. Gemini 평균 품질
  변화는 작지만 합성 두 케이스에서 소폭 하락했으므로 분포 민감성은 남아 있다.
- `boundary_and_noise`의 100개 truth noise를 baseline/current 모두 0개로 판정했다.
  이는 이번 최적화 회귀가 아니라 기존 noise 판정의 일반화 한계다. 노이즈 품질을
  제품 요구사항으로 삼는다면 별도의 labeled-noise 평가와 알고리즘 개선이 필요하다.
- 합성셋은 케이스당 360~400건, 한 생성 seed뿐이다. 더 강한 주장을 하려면 생성
  seed·차원·불균형 비율을 sweep하고 실제 도메인 외 데이터도 추가해야 한다.
- 이번 수치는 fit/skip-visualization 중심이다. update, visualization, 장기 state
  누적 성능은 이 리뷰의 범위가 아니다.
- baseline→current는 여러 커밋의 묶음 비교다. 개별 커밋 인과는 기존 microbenchmark와
  commit별 기록을 사용해야 하며 이 종합 측정만으로 분해할 수 없다.

현재 main 최적화는 유지할 근거가 충분하다. 운영에서 정확한 계층 path 재현이
필수라면 exact 모드 또는 보수 설정을 별도 기준으로 유지하고, fast 경로에는
ARI/NMI 및 작은 군집 recall 회귀 기준을 CI 벤치마크로 두는 것이 안전하다.

