# Wikipedia 클러스터링 벤치마크 실험 과정과 결과

이 문서는 2026-08-23에 이미 생성된 벤치마크 보고서와 산출물만 모아 정리한 기록이다. 이 문서를 만들기 위해 벤치마크를 다시 실행하지 않았으며, 코드나 기존 생성 산출물도 수정하지 않았다. 표의 수치는 원본 JSON의 값을 초 단위·KiB 단위로 표시하고 필요한 경우 소수점 셋째 자리에서 반올림했다.

## 1. 실험 목적과 판정 원칙

비교 대상은 PCA 공간에서 이웃을 찾는 exact-kNN 기반 HDBSCAN과 hierarchical FCM이다. PyNNDescent ANN은 exact-kNN을 대체할 수 있는지 별도로 검증했다.

품질은 복제하지 않은 원본 Wikipedia 720개에서만 판정했다. 원본 데이터를 split 내부에서 반복해 10,000개로 만든 실험은 실행시간·메모리 스케일링 관찰용이며, 클러스터 품질 결론이나 ANN 채택 판단에 사용하지 않았다. 실제 10,000개 ANN 보고서도 `quality_eligible: false`, `quality_use_prohibited: true`로 기록되어 있다.

최종 정책은 다음과 같다.

- 기본 이웃 검색: exact-kNN
- PyNNDescent: 품질·성능 실험용 선택지로 보존
- ANN 품질 통과 기준: Leaf NMI 하락 최대 0.01 이하, 추천 leaf 일치율 최소 0.95 이상
- 10,000개 결과: 품질이 아니라 비용과 병목 분석에만 사용

## 2. 공통 실험 방법

원본 720개 품질 벤치마크는 discovery 432개, calibration 144개, test 144개로 나눴다. seed `42, 43, 44, 45, 46`을 순차 실행했고, 각 실행에서 PCA 차원과 이웃 수를 자동 선택한 뒤 calibration으로 HDBSCAN 설정을 고르고 test에서 평가했다. exact-kNN 대 FCM 보고서의 프로토콜에는 `sequential: true`, 자동 PCA·neighbor-k·FCM cluster-k 탐색이 기록되어 있다.

시간·메모리 스케일링은 원본 720개의 동일 split 비율(0.6/0.2/0.2)을 split 내부에서 반복해 표본 수를 `100, 178, 316, 562, 1,000, 1,778, 3,162, 5,623, 10,000`으로 만들었다. 모든 점에서 exact와 PyNNDescent를 별도 프로세스로 측정했으며, CPU 5에 고정한 단일 코어 조건과 다음 1-thread 환경을 사용했다.

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMBA_NUM_THREADS=1
```

ANN 설정은 Euclidean metric, graph neighbors 32, query epsilon 0.1, `random_state=42`, `n_jobs=1`이다. ANN 인덱스 구축·JIT warm-up·calibration query·test query를 분리 기록했고, exact와 ANN을 동시에 실행해 자원을 경쟁시키지 않았다. 따라서 scaling 시간은 `neighbor_search_total_sec`(인덱스 구축 + query, warm-up 제외)를 주 지표로 사용한다.

## 3. 원본 720개: exact-kNN과 FCM 품질

원본 720개, 5개 seed의 aggregate 결과는 다음과 같다. exact-kNN 값은 [exact-kNN 대 FCM 보고서](wikipedia-hdbscan-knn-vs-fcm-2026-08-23/report.json)의 `hdbscan_knn_*` 항목이며, FCM 값은 같은 JSON의 `fcm_*` 항목이다.

| 지표 | HDBSCAN + exact-kNN | hierarchical FCM | 차이(HDBSCAN−FCM) |
|---|---:|---:|---:|
| Leaf NMI | 0.788858 | 0.738747 | +0.050111 |
| Leaf ARI | 0.609111 | 0.558149 | +0.050962 |
| 계층 거리(낮을수록 좋음) | 0.377778 | 0.423611 | −0.045833 |

따라서 원본 품질 기준에서는 exact-kNN HDBSCAN이 FCM보다 Leaf NMI·ARI가 높고 계층 거리도 낮다. 이 결과는 10,000개 복제 실험의 품질 수치와 섞지 않았다.

## 4. 원본 720개: ANN 품질 검증

[ANN 품질 보고서](wikipedia-ann-quality-720-2026-08-23/report.json)는 같은 원본 720개 split과 seed 5개에서 exact와 PyNNDescent를 비교한다. ANN aggregate는 다음과 같다.

| 항목 | 결과 | 기준 | 판정 |
|---|---:|---:|---|
| exact 평균 Leaf NMI | 0.791358 | — | — |
| PyNNDescent 평균 Leaf NMI | 0.790470 | — | — |
| 평균 NMI 차이(exact−ANN) | 0.000889 | — | 참고 |
| seed별 최대 Leaf NMI 하락 | 0.008049 | ≤ 0.010000 | 통과 |
| 추천 leaf 최소 일치율 | 0.993056 (99.31%) | ≥ 0.950000 (95%) | 통과 |

보고서의 `acceptance.passed`는 `true`다. 즉 원본 720개 품질만 보면 ANN은 exact를 대체할 수 있는 품질 조건을 통과했다. 다만 아래 성능 결과처럼 현재 표본 규모에서는 ANN 인덱스 구축과 고정 메모리 비용이 크므로, 품질 통과만으로 기본 backend를 ANN으로 바꾸지는 않는다.

## 5. 이웃 검색 시간·메모리 스케일링

전체 9개 표본 수의 결과는 [scaling JSON](knn-ann-scaling-100-to-10000-2026-08-23/report.json)에 있고, 같은 데이터를 [CSV](knn-ann-scaling-100-to-10000-2026-08-23/scaling.csv)와 [그래프](knn-ann-scaling-100-to-10000-2026-08-23/scaling.png)로도 제공한다. 다음 표는 JSON의 `neighbor_search_total_sec`와 `peak_rss_delta_kib`를 옮긴 것이다. 시간은 인덱스 구축과 query의 합이며 ANN JIT warm-up은 제외했다. RSS는 실행 직전 baseline 대비 peak 증가량이다.

| 표본 수 | exact 시간(s) | ANN 시간(s) | exact peak RSS 증가(KiB) | ANN peak RSS 증가(KiB) |
|---:|---:|---:|---:|---:|
| 100 | 0.011 | 19.491 | 5,044 | 430,708 |
| 178 | 0.011 | 19.457 | 6,168 | 433,092 |
| 316 | 0.012 | 19.435 | 8,180 | 437,416 |
| 562 | 0.012 | 19.564 | 11,176 | 443,488 |
| 1,000 | 0.017 | 19.433 | 15,140 | 450,356 |
| 1,778 | 0.023 | 19.506 | 22,612 | 470,576 |
| 3,162 | 0.041 | 19.693 | 34,752 | 504,760 |
| 5,623 | 0.094 | 20.255 | 61,424 | 478,160 |
| 10,000 | 0.251 | 20.080 | 104,380 | 516,492 |

10,000개에서 query만 보면 exact 0.249441초, ANN 0.109661초로 ANN이 빠르다. 그러나 인덱스 구축까지 포함하면 exact 0.251212초, ANN 20.079886초이며, ANN의 JIT warm-up까지 포함한 값은 31.196331초다. 같은 조건에서 ANN은 10,000개까지 전체 neighbor search와 peak RSS 모두 exact보다 크다. 이 때문에 기본 backend는 exact-kNN으로 유지한다.

## 6. 10,000개 HDBSCAN: 최적화 전후와 FCM

10,000개 데이터는 원본 split 내부 반복으로 만든 성능 측정용 데이터다. 최적화 전 [기존 10k 보고서](wikipedia-hdbscan-knn-vs-fcm-10000-2026-08-23/report.json)와 최적화 후 [projection/state 재사용 보고서](wikipedia-hdbscan-exact-vs-fcm-10000-reuse-2026-08-23/report.json)를 비교했다.

| 경로 | 최적화 전(s) | 최적화 후(s) |
|---|---:|---:|
| HDBSCAN 전체 | 368.799404 | 119.470509 |
| HDBSCAN 자동 PCA | 12.136678 | 14.526100 |
| HDBSCAN calibration | 326.011114 | 100.378167 |
| HDBSCAN final fit/test | 30.651612 | 4.566243 |
| FCM 전체 | 75.458021 | 88.983818 |

최적화 후 HDBSCAN은 calibration에서 선택한 state를 final 단계에서 재사용했으며, 보고서의 `hdbscan_final_selected_state_reused`와 stage의 `reused_from_calibration`이 모두 `true`다. HDBSCAN 전체 시간은 약 3.087배(약 67.6%) 줄었다. FCM 수치는 두 별도 실행의 관측값이므로 전후 차이는 환경·실행 변동을 포함하며, projection/state 재사용 효과의 비교 대상은 HDBSCAN 단계다. 최적화 후 관측된 전체 시간은 HDBSCAN 119.470509초, FCM 88.983818초다.

10,000개 품질 지표는 복제 데이터의 특성 때문에 이 문서의 품질 결론에 사용하지 않는다. 성능 비교에서만 보면 최적화 후에도 HDBSCAN이 FCM보다 약 1.343배 길다.

## 7. 최적화 후 10,000개 단계별 시간과 RSS

상세 값은 [최적화 후 seed-42 run JSON](wikipedia-hdbscan-exact-vs-fcm-10000-reuse-2026-08-23/seed-42/run.json)의 `timing`에 있다. 시간은 초, RSS는 KiB다.

| 단계 | 시간(s) | baseline RSS | peak RSS | peak 증가 |
|---|---:|---:|---:|---:|
| HDBSCAN 자동 PCA | 14.526100 | 489,532 | 1,070,100 | 580,568 |
| HDBSCAN calibration 전체 | 100.378167 | 564,880 | 897,012 | 332,132 |
| └ PCA fit/transform | 0.692638 | — | — | — |
| └ exact-kNN index build | 0.013614 | — | — | — |
| └ UMAP fit | 44.788760 | — | — | — |
| └ calibration PCA transform | 0.005995 | — | — | — |
| └ calibration UMAP transform | 14.548289 | — | — | — |
| └ calibration neighbor query | 0.133082 | — | — | — |
| └ HDBSCAN calibration 후보 평가 | 40.080837 | — | — | — |
| 선택 state 재사용 | 0.000000 | — | — | — |
| test PCA transform | 0.005961 | 897,012 | 897,012 | 0 |
| test UMAP transform | 1.718212 | 897,012 | 897,012 | 0 |
| test neighbor query | 0.108988 | 897,012 | 897,012 | 0 |
| test prediction | 2.733082 | 897,012 | 897,012 | 0 |
| HDBSCAN 전체 | 119.470509 | — | — | — |
| FCM fit | 88.829934 | 897,012 | 1,427,004 | 529,992 |
| FCM test assignment | 0.153884 | 868,032 | 885,748 | 17,716 |
| FCM 전체 | 88.983818 | — | — | — |
| 시각화 | 0.346463 | 885,748 | 888,436 | 2,688 |

최적화 후 HDBSCAN calibration 안에서 가장 큰 시간 항목은 UMAP fit 44.788760초와 HDBSCAN 후보 평가 40.080837초다. exact-kNN index build와 query는 각각 0.013614초와 0.133082초로 작다. 따라서 현재 남은 주요 병목은 이웃 검색이 아니라 UMAP과 HDBSCAN calibration 평가다. RSS peak는 자동 PCA 단계 1,070,100 KiB, FCM fit 단계 1,427,004 KiB로 관측됐다.

## 8. 결론

1. 원본 720개 품질에서는 exact-kNN HDBSCAN이 hierarchical FCM보다 Leaf NMI 0.050111, Leaf ARI 0.050962 높고 계층 거리도 0.045833 낮았다.
2. PyNNDescent는 원본 720개에서 최대 NMI 하락 0.008049, 추천 leaf 최소 일치율 99.31%로 ANN 품질 기준을 통과했다.
3. 하지만 100~10,000개 스케일링에서는 ANN 인덱스 구축·JIT와 메모리 고정 비용 때문에 전체 이웃 검색은 exact보다 느리고 RSS도 컸다.
4. calibration projection과 선택 state 재사용으로 10,000개 HDBSCAN 시간이 368.799404초에서 119.470509초로 줄었지만, 관측된 FCM 88.983818초보다는 길었다.
5. 따라서 현재 운영 기본값은 exact-kNN으로 두고 ANN은 실험용으로 유지한다. 다음 최적화 대상은 UMAP과 HDBSCAN calibration 평가다.

## 9. 출처 보고서와 산출물

| 내용 | 보고서 | 보조 산출물 |
|---|---|---|
| 원본 720개 exact-kNN 대 FCM 품질 | [report.json](wikipedia-hdbscan-knn-vs-fcm-2026-08-23/report.json) | [runs.csv](wikipedia-hdbscan-knn-vs-fcm-2026-08-23/runs.csv), seed별 `run.json`·comparison.png |
| 원본 720개 exact 대 PyNNDescent 품질 | [report.json](wikipedia-ann-quality-720-2026-08-23/report.json) | [runs.csv](wikipedia-ann-quality-720-2026-08-23/runs.csv) |
| 100~10,000개 exact 대 ANN scaling | [report.json](knn-ann-scaling-100-to-10000-2026-08-23/report.json) | [scaling.csv](knn-ann-scaling-100-to-10000-2026-08-23/scaling.csv), [scaling.png](knn-ann-scaling-100-to-10000-2026-08-23/scaling.png) |
| 10,000개 ANN 성능·품질 비사용 표기 | [report.json](wikipedia-ann-scaling-10000-2026-08-23/report.json) | — |
| 10,000개 HDBSCAN 대 FCM, 최적화 전 | [report.json](wikipedia-hdbscan-knn-vs-fcm-10000-2026-08-23/report.json) | [run.json](wikipedia-hdbscan-knn-vs-fcm-10000-2026-08-23/seed-42/run.json), comparison.png |
| 10,000개 HDBSCAN 대 FCM, projection/state 재사용 후 | [report.json](wikipedia-hdbscan-exact-vs-fcm-10000-reuse-2026-08-23/report.json) | [run.json](wikipedia-hdbscan-exact-vs-fcm-10000-reuse-2026-08-23/seed-42/run.json), comparison.png |

