# 지표 전용 클러스터링 평가기

`extract_clustering_metrics.py`는 저장된 할당 CSV를 읽으며 PCA, FCM
또는 다른 클러스터링 모델을 적합하지 않는다. 한 번의 호출로 하나의 조건이나
여러 조건을 평가할 수 있다.

```bash
./.venv/bin/python extract_clustering_metrics.py \
  --assignments \
    /path/to/assignments_content_only.csv \
    /path/to/assignments_content_plus_tag.csv \
  --features \
    /path/to/content_features.npy \
    /path/to/content_plus_tag_features.npy \
  --output-csv clustering_metrics.csv
```

`--features`는 선택 사항이다. 지정하지 않아도 외부 지표와 저장된 소속도
지표를 계산할 수 있다. 지정하면 해당 특성 공간에서 실루엣과 XB를
계산한다. PCA 이후 지표를 정확히 재현하려면 저장된 PCA 이후 특성 행렬을
전달한다. 임베딩 JSON을 특성 원천으로 사용할 수도 있으며, 이 경우 원시
임베딩 공간에서 평가한다.

기존 메모리 내 파이프라인은 적합된 중심을 동일한 지표 핵심 로직에 전달하므로
XB와 퍼지 실루엣 값에 정확한 모델 중심을 사용한다. 독립 실행형 CSV 도구는
중심을 별도로 제공하지 않는 한 저장된 소속도에서 중심을 유도한다.

할당 CSV에는 외부 평가를 위한 `class`와 `class_hierarchy` 열, 그리고
PC, 수정 PC, PE, 정규화 PE 계산을 위한 `membership_0`, `membership_1`,
... 열을 포함할 수 있다. 대상 열이 없으면 `--metadata-json`을 사용해
`id` 기준으로 정렬한다(ID가 없으면 행 순서를 사용한다).

지표의 방향:

- NMI, ARI, 실루엣, PC: 높을수록 좋다.
- XB, PE, 정규화 PE: 낮을수록 좋다.

이 도구는 `fits_clustering_model: false`를 보고하며, 적합 과정에서 정답
라벨을 사용하지 않는다.
