# 빠른 FCM 파이프라인

현재 `main`의 `--fast`는 증분 적합 명령에서 범위가 제한된 scout/refine 경로를
활성화한다. 일반 경로는 최종 검증용으로 유지하며, fast 결과를 확정할 때는 같은
입력·seed에서 일반 경로와 후보 지표를 비교한다.

```bash
./.venv/bin/python incremental_clustering.py fit \
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
./.venv/bin/python incremental_clustering.py fit \
  --input-json /workspaces/codespaces-blank/dbpedia_gemini_embeddings.json.gz \
  --state-output results/fast_scout.state.pkl \
  --assignments-output results/fast_scout_assignments.csv \
  --tree-output results/fast_scout_tree.json \
  --pca-components 192 \
  --fast \
  --skip-visualization
```

`--fast`는 각 노드에서 최대 1,000개를 K 탐색용 표본으로 추출하고, 기본 후보
`1.2, 1.4, 1.6, 1.8, 2.0` 중 안정적인 첫 `m`을 선택한다. 안정적인 부모의 `m`은
자식에서 재사용하며 K scout가 불안정할 때만 다시 찾는다. scout 점수가 좋은 상위
두 K만 전체 노드에서 정밀화하고, 점수 차가 충분히 크면 하나만 정밀화한다.
`--skip-visualization`은 클러스터링 전용 상태를 생성하므로, 증분 업데이트에
사용하기 전에 해당 옵션 없이 다시 적합해야 한다.

로드된 데이터셋에서 재현 가능한 무작위 부분집합으로 적합하려면
`--dataset-sample-size`를 추가하고, 필요하면 `--dataset-sample-seed`도
지정한다.

```bash
./.venv/bin/python incremental_clustering.py fit \
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
