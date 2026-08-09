# `.mds` 문서 안내

현재 알고리즘의 기준 문서는 [`CURRENT_ALGORITHM.md`](CURRENT_ALGORITHM.md)다.
`incremental_clustering.py fit/update`의 자동 PCA 선택, 재귀 구면 FCM,
`multi_metric` K 선택, 노이즈 판정, 약지도 UMAP, 온라인 중심 갱신과 고정 좌표
증분 업데이트를 설명한다.

이 디렉터리의 나머지 문서는 과거 실험·설계 결정·비교를 보존하기 위한 자료다.
현재 기본 경로는 문서마다 다를 수 있으므로 새 구현이나 운영 설정을 확인할
때는 최신 기준 문서를 우선한다.

## 보존된 과거 자료

- `PROJECT_DECISION_HISTORY.md`: 프로젝트 의사결정과 과거 성능 비교
- `PCA_DIMENSION_SELECTION.md`: 고정 PCA 후보 평가와 자동 선택 근거
- `TAG_EMBEDDING_EXPERIMENT.md`: 태그 임베딩 실험
- `hierarchical_clustering_js_plan.md`: JavaScript/TypeScript 이식 기획
- `JS_PERFORMANCE_OPTIMIZATION_PLAN.md`: Python 알고리즘 최적화 후 JS 이식 계획
- `PYTHON_PERFORMANCE_BACKLOG.md`: 기존 계획 밖 Python 성능 후보와 작업 트래커
- `MAIN_OPTIMIZATION_REVIEW.md`: main 최적화의 종합 성능·품질·과최적화 검증
- `incremental_clustering.md`: 증분 사용법의 초기 문서
- `FAST_FCM.md`: 빠른 FCM 탐색의 초기 사용 예
- `METRIC_EXTRACTOR.md`: 지표 전용 평가 도구 설명
- `next_tesk.md`: 미완료 작업 메모

이 디렉터리는 의도적으로 Git에서 무시되므로 현재 작업 트리에만 남고, 새로
만드는 Git 작업 트리에는 체크아웃되지 않는다.
