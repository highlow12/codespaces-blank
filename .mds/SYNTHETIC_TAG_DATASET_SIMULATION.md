# 합성 태그 데이터셋 생성 및 3,000개 시뮬레이션

이 문서는 태그를 노트의 정답 라벨로 직접 사용하지 않고, 숨은 concept
공간을 노트 임베딩과 관측 태그가 각각 불완전하게 관측하는 상황을 만들기
위한 합성 데이터 생성 방법과 첫 시뮬레이션 결과를 기록한다.

## 1. 생성 원칙

숨은 concept 공간을 평가 기준으로 두고 다음 두 경로를 독립적으로 생성한다.

```text
random root concepts
  -> correlated hierarchical concepts
  -> Dirichlet soft semantic membership
  -> semantic vector + note-specific component + embedding noise
  -> note embedding

same hidden concepts
  -> tag selection + missing tags + generic tags + wrong tags + extra tags
  -> observed multi-tags
```

태그 임베딩은 클러스터 정답으로 전달하지 않는다. 클러스터링에는 콘텐츠
임베딩 또는 콘텐츠와 관측 태그를 결합한 특징만 전달하고, 숨은 concept
membership은 결과 평가에만 사용한다.

## 2. 숨은 계층 concept 생성

초기 root concept는 10개의 완전 랜덤 768차원 벡터를 생성한 뒤 행별 L2
정규화한다. 이후 다음 네 단계의 계층을 만든다.

| 단계 | concept 수 | 생성 방식 |
|---|---:|---|
| root | 10 | 랜덤 고차원 벡터 |
| level 1 | 20 | 부모 2~4개를 Dirichlet(`alpha=0.7`) 가중 합산 |
| level 2 | 40 | level 1 부모 2~4개를 같은 방식으로 합산 |
| level 3 | 80 | level 2 부모 2~4개를 같은 방식으로 합산 |

각 하위 concept에는 작은 방향 노이즈 `0.03`을 추가한 뒤 다시 L2
정규화한다. 동시에 부모의 root membership도 같은 가중치로 합산해 각
concept의 숨은 root profile을 보존한다.

## 3. 노트 임베딩 생성

각 노트는 2~3개의 root concept를 선택하고 Dirichlet(`alpha=0.8`) 가중치를
샘플링한다. 각 hierarchy level에서는 이 root profile과 관련성이 높은
concept 중 2~4개를 선택하고 Dirichlet(`alpha=0.7`) 가중치를 적용한다.

네 level의 기여도 자체도 Dirichlet(`alpha=2.0`)에서 샘플링한다. 따라서
노트는 태그의 단순 평균이 아니라 여러 계층 concept의 가중 혼합이 된다.

```text
semantic = sum(level_weight * weighted_level_concept)
embedding = normalize(
    semantic
    + 0.15 * note_specific_component
    + 0.20 * embedding_noise
)
```

`note_specific_component`와 `embedding_noise`는 각각 독립 랜덤 방향 벡터다.
최종 노트 임베딩과 모든 중간 벡터는 방향 기반 비교를 위해 L2 정규화한다.

## 4. 관측 다중 태그 생성

각 노트의 의도된 태그는 root를 제외한 level 1, 2, 3에서 semantic 혼합
기여도가 가장 큰 concept 하나씩을 선택한다. 즉 기본 의미는 세 개의 서로
다른 태그로 구성된다.

태그 임베딩은 해당 concept 벡터에 태그 전용 방향 노이즈 `0.05`를 더한 뒤
정규화한다. 사용자에게 관측되는 태그에는 다음 변형을 적용한다.

| 변형 | 확률 | 동작 |
|---|---:|---|
| 누락 | 0.15 | 의도된 태그를 관측 목록에서 제거 |
| 오태그 | 0.10 | 같은 level의 다른 concept로 교체 |
| 일반 태그 | 0.20 | 관련 root concept 태그를 추가 |
| 과잉 태그 | 0.25 | 임의 level의 추가 태그를 부착 |

관측 태그가 모두 사라지는 경우에는 의도된 태그 하나를 fallback으로
남긴다. 노트별 관측 태그 임베딩은 여러 태그 벡터를 합산한 뒤 L2
정규화한다. 따라서 다중 태그의 의미를 하나의 태그 블록으로 전달하지만,
태그 개수 자체가 콘텐츠 벡터의 크기를 직접 키우지는 않는다.

## 5. 3,000개 시뮬레이션 설정

- seed: `20260815`
- 노트 수: `3,000`
- embedding 차원: `768`
- root concept 수: `10`
- 고정 clustering PCA: `96`
- FCM cluster 수: `10`
- 시각화: 실행하지 않음
- 평가 기준: 숨은 10개 root에 대한 soft membership과 그 dominant root
- dominant root는 평가용 투영이며 clustering 입력에는 전달하지 않음

비교 조건은 콘텐츠만, 태그만, 콘텐츠와 태그 결합 `weight=1.0/0.75/0.5`,
그리고 태그 노트 정렬을 무작위로 섞은 `weight=0.5` 대조군이다.

## 6. 결과

| 조건 | ARI | NMI | soft root alignment | silhouette |
|---|---:|---:|---:|---:|
| 콘텐츠만 | **0.5880** | **0.6272** | **0.1773** | **0.1654** |
| 태그만 | 0.1539 | 0.1896 | 0.1390 | 0.1582 |
| 콘텐츠 + 태그 `1.0` | 0.2895 | 0.3506 | 0.1592 | 0.1017 |
| 콘텐츠 + 태그 `0.75` | 0.3676 | 0.4349 | 0.1668 | 0.1034 |
| 콘텐츠 + 태그 `0.5` | 0.5310 | 0.5697 | 0.1754 | 0.1344 |
| 콘텐츠 + 섞은 태그 `0.5` | 0.4727 | 0.5532 | 0.1735 | 0.0918 |

이번 생성 규칙에서는 콘텐츠만 사용한 결과가 가장 좋았다. 관측 태그에
누락·오태그·일반·과잉 태그가 동시에 들어가므로 높은 태그 가중치는 숨은
semantic 구조를 흐렸다. `weight=0.5`는 태그 결합 조건 중 가장 안정적이며,
정렬된 태그가 무작위로 섞은 태그보다 ARI 약 `0.058`, NMI 약 `0.017`
높았다. 따라서 관측 태그에 latent concept 신호가 일부 남아 있다는 점은
확인했지만, 이 데이터에서는 태그가 콘텐츠를 보완할 정도로 깨끗하지는
않았다.

이 결과는 한 seed의 초기 검증이다. 일반화 결론을 내리려면 seed를 바꾸고,
태그 오류율·누락률·노트별 태그 수·콘텐츠 노이즈를 독립적으로 변화시키는
추가 실험이 필요하다.

## 7. 산출물

실험 산출물은 현재 실행 환경의 `/tmp`에 저장했다.

- `synthetic_notes_3000.npz`: 임베딩, 관측 태그 임베딩, 숨은 root membership
- `synthetic_notes_3000.json.gz`: 노트별 embedding과 true/observed tag 메타데이터
- `report_pca96.json`: PCA-96 비교 결과

