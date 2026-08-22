# Agent instructions

## Search tools

- For text and file searches in this workspace, always use `./.local/bin/rg` (or the absolute path `/workspaces/codespaces-blank/.local/bin/rg`) rather than relying on `rg` being available on `PATH`.
- Use `./.local/bin/rg --files` when listing tracked or unignored files, and `./.local/bin/rg <pattern>` when searching file contents.

## Python environment

- Use the repository virtual environment at `./.venv/bin/python` for scripts, imports, and tests.
- If it does not exist, create it with `python3 -m venv .venv`.
- Install dependencies with `./.venv/bin/python -m pip install -r requirements.txt`.
- Run the test suite with `./.venv/bin/python -m unittest discover -s . -p 'test_*.py'`.

## Sub-agent implementation

- When the primary agent is running a Sol model, spawn a `gpt-5.6-luna` sub-agent with `high` reasoning effort and assign it the actual implementation and test execution.
- Give the Luna sub-agent a concrete, self-contained task, require it to edit the workspace directly, and review its changes and test results before reporting completion.

## Clustering validation data

- Do not use `dbpedia_label_embeddings.json` for clustering or incremental-clustering validation. It contains only 18 tag-label embeddings and is not the clustering test dataset.
- Use the 3,000-record Gemini embedding dataset, `dbpedia_gemini_embeddings.json` or its gzip form `dbpedia_gemini_embeddings.json.gz`, for clustering tests and benchmarks.
- For a quick run on the Gemini dataset, limit the rows with `--dataset-sample-size` (and optionally `--dataset-sample-seed`) and add `--fast`. Do not substitute the tag-only dataset when a smaller run is needed.

Example:

```bash
./.venv/bin/python incremental_clustering.py fit \
  --input-json dbpedia_gemini_embeddings.json.gz \
  --dataset-sample-size 100 \
  --dataset-sample-seed 42 \
  --fast \
  --state-output /tmp/incremental-fast.state.pkl \
  --skip-visualization
```

## Parallel independent runs

- Run independent benchmarks, seeds, or dataset slices in parallel whenever practical to reduce elapsed time.
- Limit concurrency based on available CPU and memory; clustering runs may be resource-intensive, so avoid oversubscribing the machine.
- Keep each run's output in a separate directory and aggregate results only after all parallel jobs complete.
