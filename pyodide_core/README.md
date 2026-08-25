# Pyodide core

`pyodide_core/atomic_clustering` is the first browser-oriented extraction of
the default clustering path:

```text
normalized embeddings -> PCA(auto) -> UMAP -> HDBSCAN leaves
                     -> membership-weighted bottom-up hierarchy
```

The public function is `cluster_documents`. It returns only plain Python
dictionaries, lists, strings, and numbers, so a Pyodide caller can pass the
result directly through `toJs()`:

```javascript
const py = await loadPyodide();
py.runPython("import sys; sys.path.insert(0, '/app/pyodide_core'); from atomic_clustering import cluster_documents");
// A small worker wrapper should convert JsProxy arrays to lists/NumPy arrays
// and pass keyword-only arguments explicitly.
const result = py.runPython(`cluster_documents(embeddings, ids=ids, config=config)`, {
  locals: py.toPy({embeddings, ids, config}),
});
const jsResult = result.toJs({dict_converter: Object.fromEntries});
result.destroy();
```

In practice, expose a tiny Python wrapper in the worker so JavaScript arrays
are converted to NumPy arrays and the callback boundary is easier to manage.
The equivalent Python call is:

```python
from atomic_clustering import cluster_documents
result = cluster_documents(embeddings, ids=ids, config={...})
```

Standard Pyodide does not provide `umap-learn` and `hdbscan` in its normal
package set. `dependency_status()` reports the runtime situation, and a
browser/JavaScript implementation can be injected with `discovery_runner`:

```python
def discover(pca_features, config):
    return {
        "umap_features": ...,       # shape (n, d)
        "leaf_labels": ...,         # -1, 0, ..., C-1
        "memberships": ...,         # shape (n, C)
        "probabilities": ...,       # optional shape (n,)
        "outlier_scores": ...,      # optional shape (n,)
    }

result = cluster_documents(embeddings, discovery_runner=discover)
```

This package intentionally has no pandas, CLI, filesystem, plotting, or
dataset dependencies. The fitted sklearn PCA object is used internally but
is not returned in the result, which keeps the API JSON/JS-friendly.
