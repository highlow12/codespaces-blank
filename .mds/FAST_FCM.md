# 빠른 FCM 파이프라인

`fast-fcm-pipeline` 브랜치는 증분 적합 명령에 범위가 제한된 거친 단계에서
정밀 단계로 이어지는 경로를 추가한다. `--fast`로 활성화하며, 일반 후보
선택기를 기본값으로 유지한다.

```bash
python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json.gz \
  --state-output results/fast.state.pkl \
  --assignments-output results/fast_assignments.csv \
  --coordinates-output results/fast_coordinates.csv \
  --tree-output results/fast_tree.json \
  --plot-output results/fast_scatter.png \
  --pca-components 192 \
  --max-depth 4 \
  --max-clusters 8 \
  --fast
```

클러스터링만 반복해서 실험하려면 UMAP을 건너뛰고 최종 후보에 대해서만 한 번
적합한다.

```bash
python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json.gz \
  --state-output results/fast_scout.state.pkl \
  --assignments-output results/fast_scout_assignments.csv \
  --tree-output results/fast_scout_tree.json \
  --pca-components 192 \
  --fast \
  --skip-visualization
```

`--fast`는 각 노드에서 K 탐색용 표본을 추출하고, 기본 후보
`1.2, 1.4, 1.6, 1.8, 2.0` 순서로 `m`을 시험하며,
가장 좋은 K만 정밀화하고, 안정성이 목표보다 낮을 때만 재시작 횟수를 늘린다.
`--skip-visualization`은 클러스터링 전용 상태를 생성하므로, 증분 업데이트에
사용하기 전에 해당 옵션 없이 다시 적합해야 한다.

로드된 데이터셋에서 재현 가능한 무작위 부분집합으로 적합하려면
`--dataset-sample-size`를 추가하고, 필요하면 `--dataset-sample-seed`도
지정한다.

```bash
python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 500 \
  --dataset-sample-seed 2026 \
  --state-output results/sample500.state.pkl \
  --pca-components 192 \
  --fast
```

복원 없이 표본을 추출하며 문서 ID와 메타데이터를 보존한다. 표본 추출 시드는
기본적으로 `--seed`를 사용한다. 이는 데이터셋 수준의 표본이며, 빠른
클러스터링 알고리즘 내부에서 노드별 K 탐색을 제어하는
`--fast-sample-size`와는 별개다.
