# Synthetic tag fusion results — 2026-08-09

## 실행 범위

전체 sweep은 3개 seed, content noise 5개, tag corruption multiplier 5개,
tag weight 4개를 사용했다. 총 1,575개 run을 저장했다.

- content noise: `0.05, 0.10, 0.20, 0.30, 0.40`
- tag corruption: `0, 0.5, 1, 1.5, 2`
- tag weight: `0.25, 0.5, 1, 2`
- seeds: `42, 43, 44`
- fixed root K: `10`
- 각 셀의 평가는 content-only, observed/oracle/shuffled tag, additive/concat/PCA ablation을 포함한다.

아래 delta는 같은 seed·content noise·corruption 셀의 content-only 대비 평균 변화다.
주 지표는 soft membership cosine이며, boundary는
`max(true_membership) < 0.6`인 노트만 계산했다.

## 전체 평균

| tag source / variant | weight | Δ membership cosine | Δ boundary cosine | Δ ARI |
| --- | ---: | ---: | ---: | ---: |
| observed / additive | 0.25 | +0.0703 | +0.0534 | +0.1518 |
| observed / additive | 0.50 | +0.1226 | +0.0895 | +0.1729 |
| observed / additive | 1.00 | +0.1361 | +0.0996 | +0.0807 |
| observed / additive | 2.00 | +0.1296 | +0.0938 | +0.0757 |
| observed / concat | 1.00 | +0.0882 | +0.0614 | +0.0544 |
| observed / same-PCA additive | 1.00 | +0.0799 | +0.0624 | +0.0472 |
| oracle / additive | 1.00 | +0.3218 | +0.2314 | +0.5581 |
| shuffled / additive | 1.00 | -0.0917 | -0.0796 | -0.2867 |

올바른 태그의 효과는 실제 semantic signal이며, shuffled tag의 효과와 명확히
구분된다. 이 조건에서는 additive fusion이 concat과 same-PCA additive보다 soft
membership 기준으로 좋았다.

## 전환 경계

아래는 observed additive, tag weight `1.0`의 membership cosine delta다. 각 셀은
세 seed 평균이다.

| content noise \ corruption | 0 | 0.5 | 1.0 | 1.5 | 2.0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | +0.2493 | +0.1553 | +0.0448 | -0.0155 | -0.0784 |
| 0.10 | +0.2676 | +0.1635 | +0.0956 | -0.0074 | -0.0508 |
| 0.20 | +0.3425 | +0.2322 | +0.1454 | +0.0383 | -0.0115 |
| 0.30 | +0.3667 | +0.2504 | +0.1788 | +0.0705 | +0.0328 |
| 0.40 | +0.3737 | +0.2649 | +0.1706 | +0.0881 | +0.0340 |

content noise가 낮은 경우에는 corruption multiplier `1.5–2.0`부터 additive
fusion이 손해로 전환된다. content noise가 높아지면 태그가 보완 신호가 되어
corruption `2.0`에서도 평균적으로 작은 이득이 남는다. 다만 이는 태그 품질이
좋아졌다는 뜻이 아니라, 본문 관측이 더 불확실해져 태그의 잔여 신호 가치가
상대적으로 커졌다는 뜻이다.

## 결론

1. 태그가 무의미하다는 가설은 기각된다. 올바른 태그는 content-only보다
   membership 품질을 개선했고, oracle tag에서는 개선 폭이 더 컸다.
2. 현재 corruption 수준과 weight를 모르는 상태에서 early fusion을 고정적으로
   적용하는 것은 안전하지 않다. 낮은 본문 noise에서는 오히려 손해가 시작된다.
3. 이 synthetic generator에서는 additive fusion이 가장 유망하지만, 태그 품질을
   추정해 weight를 낮추거나 tags를 별도 metadata/prior/reranking 채널로 두는
   설계가 더 보수적이다.
4. boundary 노트도 같은 경계를 보였고, corruption이 심해질수록 boundary
   membership 손해가 커졌다.

## 남은 검증

이번 결과는 fixed `K=10` synthetic benchmark다. 다음 단계는 유망한 조건을
hierarchical/K-unknown 경로에서 다시 실행하고, 그 후 Gemini 데이터에서 태그를
본문 공간에 직접 결합하지 않는 metadata/prior/reranking 실험으로 넘어가는 것이다.
