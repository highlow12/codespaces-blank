# DBpedia 임베딩 클러스터링

DBpedia 문서 임베딩을 대상으로 **Spherical Fuzzy C-Means(SFCM)** 기반 클러스터링과 증분 업데이트를 실험·운영하는 프로젝트입니다. 현재 기본 경로는 본문 임베딩을 PCA로 축소한 뒤 SFCM으로 군집화하는 방식입니다.

## 현재 상태

| 영역 | 상태 | 요약 |
| --- | --- | --- |
| 증분 계층형 클러스터링 | 완료 · 통합됨 | ID 기반 추가/교체, 선택적 membership 갱신, 드리프트 기반 재클러스터링을 지원합니다. |
| 상태 저장 및 재실행 안전성 | 완료 | batch ID 멱등성, checksum envelope, atomic save, 동시 update 직렬화, 이전 상태 호환성을 갖춥니다. |
| 성능 기준선 | 측정 완료 | 3,000건 Gemini 임베딩에서 fast 모드 기준 fit 77.7초, 일반 update 5.43초를 기록했습니다. |
| 태그 융합 | 검증 전 | 현 태그 품질에서는 early fusion이 표현 공간을 왜곡할 가능성이 있어 기본 경로에 넣지 않았습니다. 합성 데이터 실험으로 유효 조건을 검증할 계획입니다. |

### 현재 결론

- 기본 설계는 `content → PCA → SFCM`입니다.
- 태그는 곧바로 임베딩에 합치지 않고, 검증 전까지 metadata·prior·reranking 후보 채널로 분리합니다.
- 증분 업데이트는 전체 문서를 매번 다시 계산하지 않습니다. 중심 이동의 영향이 충분한 문서와 새로 추가·수정된 문서만 membership을 갱신하고, 자연 noise 또는 XB 품질 저하가 감지될 때 전체 재클러스터링을 수행합니다.

## 선택의 타임라인

프로젝트에서 지금까지 선택한 방향과 그 판단 근거입니다.

| 시점 | 선택 | 이유와 현재 반영 상태 |
| --- | --- | --- |
| 2026-08-02 | 하드·평면 군집 대신 **계층형 구면 FCM** 채택 | 문서는 여러 주제에 걸칠 수 있고 주제에는 상·하위 구조가 있으므로, 하나의 하드 라벨보다 소프트 소속도와 재귀 분할을 보존하기로 했습니다. 입력과 중심을 단위 구면에 두어 의미적 방향을 기준으로 계산합니다. |
| 2026-08-02 | 군집화와 2차원 시각화를 분리 | UMAP에서 잘 갈라져 보인다고 해서 의미 군집이 좋은 것은 아니므로, 군집은 고차원 PCA 공간에서 수행하고 UMAP은 결과를 보여 주는 역할로 한정했습니다. |
| 2026-08-03 | 지역 소속도를 조건부 경로 소속도로 저장 | 서로 다른 부모 노드의 소속도를 직접 비교하지 않고, 부모까지의 확률을 곱한 경로 확률로 비교 가능하게 만들었습니다. 현재 이 상세 출력은 필요한 경우에만 opt-in합니다. |
| 2026-08-03 | 시각화에 소프트 소속도를 약하게만 반영 | 군집 라벨이 원래 임베딩 구조를 덮어쓰지 않도록 UMAP의 약지도 목표 가중치를 `0.01`로 낮게 유지했습니다. |
| 2026-08-03~04 | PCA 차원을 작업별로 분리한 뒤 자동 선택으로 전환 | 과거 고정 PCA-256(군집)·PCA-64(시각화) 비교를 바탕으로, 현재는 데이터별 k-NN 보존율이 포화되기 전의 차원을 각각 자동 선택합니다. |
| 2026-08-04 | 고정 K 대신 노드별 `multi_metric` K 선택 | 계층의 모든 노드를 같은 수로 나누지 않고, XB·실루엣·재시작 안정성·분할 계수를 함께 평가해 데이터가 지지하는 경우에만 분할합니다. |
| 2026-08-04 | 경계·이상치 문서를 별도 판정하고, 품질 악화 때만 재클러스터링 | 낮은 소속도·중심 거리·XB 악화를 함께 사용해 무조건적인 재학습과 과도한 분할을 피하도록 했습니다. |
| 2026-08-05 | 실험과 빠른 반복을 위해 `--fast` 경로와 재현 가능한 표본 추출 추가 | K 탐색용 표본, 적응적 fuzzifier, 제한된 refinement를 사용하되, 일반 선택기는 유지해 정확도와 탐색 속도를 구분했습니다. |
| 2026-08-05 | 실제 검증 데이터로 18개 태그 라벨 파일 대신 3,000건 Gemini 임베딩 사용 | 작은 태그 전용 파일은 클러스터링 품질·증분 처리 성능을 검증할 데이터가 아니므로, 표본 검증도 Gemini 데이터에서 수행하기로 했습니다. |
| 2026-08-05 | 정답 class에서 만든 태그 결합 결과를 운영 근거로 쓰지 않음 | 초기 태그 실험은 태그 신호의 상한을 확인했지만, 실제 운영 태그가 아니었습니다. 따라서 태그를 기본 경로에 편입하지 않고 별도 검증 대상으로 남겼습니다. |
| 2026-08-05 | 증분 상태를 문서별 outer product 대신 compact weight로 저장 | 상태 크기와 일반 update 비용을 낮추기 위해서입니다. 변경된 ID의 contribution만 빼고 더하는 delta update를 기본으로 채택했습니다. |
| 2026-08-06 | 전체 membership refresh 대신 중심 영향 기반의 선택 refresh | 중심이 충분히 움직였고 fuzzy weight가 큰 문서, 그리고 신규·수정 문서만 다시 계산하도록 선택했습니다. 전체 갱신은 실제 재클러스터링 때만 수행합니다. |
| 2026-08-06 | 즉시 noise 반응 대신 누적·EWMA·hysteresis·cooldown 기반 드리프트 판정 | 작은 배치의 우연한 noise로 재클러스터링이 반복되는 것을 막기 위해서입니다. |
| 2026-08-07 | flat/계층형 경로의 batch 처리 규칙을 공통 코어로 통합 | append/replace, batch ID 멱등성, replay 기록, atomic save를 동일한 방식으로 보장하기 위해서입니다. |
| 2026-08-08~09 | 수학적으로 동등한 계산 재사용과 혼합 정밀도 저장을 우선 | FCM 거리·membership 계산과 PCA 탐색의 중복을 줄이고, 임베딩·상태는 기본 `float32`, 중심·품질 통계는 `float64`로 유지해 결과를 보존하면서 CPU·RSS를 낮췄습니다. |
| 2026-08-09 | Python worker 병렬화는 보류 | 2 CPU 환경의 측정에서 순차 실행보다 느렸습니다. 따라서 복잡한 병렬화보다 입력 cache·lazy import·수치 계산 재사용을 우선합니다. |
| 현재 | 태그의 early fusion을 기본 경로에서 제외하고 `content → PCA → SFCM` 유지 | 태그 신호 자체는 확인하되, 현재 품질에서는 본문 공간의 기하를 해칠 가능성이 있습니다. 합성 데이터의 control·ablation 실험으로 이득 조건이 확인될 때만 결합 방식을 채택합니다. |

## 최근 검증 및 기준선

Gemini 임베딩 데이터셋(3,000건, 3,072차원), seed `42`, `--fast`, 시각화 포함 조건에서 측정한 값입니다. update 크기는 입력의 4%입니다.

| 입력 문서 수 | fit | 일반 update | 선택 refresh | refresh / skip | 상태 파일 | peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 26.5초 | 4.07초 | 2.32초 | 15 / 80 | 5.5 MB | 0.87 GB |
| 500 | 31.2초 | 4.34초 | 0.47초 | 86 / 404 | 25.2 MB | 0.87 GB |
| 1,000 | 39.2초 | 4.94초 | 0.37초 | 40 / 940 | 38.5 MB | 0.97 GB |
| 3,000 | 77.7초 | 5.43초 | 1.51초 | 120 / 2,820 | 88.4 MB | 1.36 GB |

증분 리팩터 통합 뒤 루트 테스트 57개와 gzip 입력 100건 fast-fit smoke test를 통과했습니다.

## 빠른 실행

의존성을 설치합니다.

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Gemini 데이터에서 빠르게 초기 상태를 적합합니다. `dbpedia_label_embeddings.json`은 태그 라벨 18개만 가진 파일이므로 클러스터링 검증에 사용하지 않습니다.

```bash
./.venv/bin/python incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --dataset-sample-seed 42 \
  --fast \
  --state-output /tmp/incremental-fast.state.pkl \
  --skip-visualization
```

시각화와 함께 flat Auto-PCA SFCM 전체 파이프라인을 실행하려면 다음을 사용합니다.

```bash
./.venv/bin/python full_pipeline.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --output-dir results/full_pipeline \
  --fast
```

증분 처리 비용을 측정하려면 다음을 사용합니다.

```bash
./.venv/bin/python benchmark_incremental_updates.py \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --update-size 10 \
  --fast \
  --output-json /tmp/incremental-benchmark.json
```

## 다음 연구 작업

태그 융합 실험은 아직 제품 경로에 반영하지 않았습니다. 다음을 통해 언제 태그가 이득이 되는지 검증합니다.

1. 상관된 root와 soft membership을 가진 합성 데이터를 생성합니다.
2. content noise, 태그 corruption, tag weight를 바꾸며 content-only·correct-tag·shuffled-tag를 비교합니다.
3. early additive/concatenation fusion, PCA 전후 결합, metadata prior·reranking을 분리해 평가합니다.
4. ARI/NMI뿐 아니라 soft membership, 특히 경계 문서의 품질을 평가합니다.

세부 가설·실험 행렬·완료 기준은 [합성 데이터 기반 태그 융합 실험 계획서](SYNTHETIC_TAG_FUSION_EXPERIMENT_PLAN.md)에 정리되어 있습니다.

## 테스트

```bash
./.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```
