# Search & Focus implementation plan

Status: planned on the PR #3 merged baseline; this document is the design and
acceptance checklist for Milestone B in [`../../plan.md`](../../plan.md).

## Goals and non-goals

Search and Focus turn the Explorer into a navigation surface without changing
cluster assignments. Search is local, deterministic, and safe to use while a
build is running. Focus changes the visible hierarchy and note list only; it
never invokes a worker or reclusters.

The current persisted schema gives us note path, title, body, mtime, result
labels/placements, generated cluster titles, and generated title keywords.
Tags and aliases are not separate persisted fields yet, so the index extracts
them from optional Markdown frontmatter when body text is available. Manual
cluster corrections are a Milestone C concern; the filter is represented as a
disabled/empty capability until that data exists rather than inventing state.

## Data and search contract

`SearchDocument` is a lightweight projection of a note and its available
cluster context:

- `path`, `title`, `body`, `tags`, and `aliases` are lexical fields;
- `clusterIds` includes the terminal leaf and its hierarchy ancestors;
- cluster titles and generated keyword scores are indexed as cluster fields;
- `mtime`, leaf label, provisional status, and manual-adjustment availability
  drive quick filters.

The parser accepts plain terms, quoted phrases, and these field qualifiers:

| Syntax | Meaning |
| --- | --- |
| `term` | all available note and cluster metadata |
| `"exact phrase"` | contiguous case-insensitive phrase |
| `tag:topic` | a parsed Markdown tag |
| `path:Projects/` | path prefix or path token |
| `cluster:research` | cluster title/keyword token |

Whitespace combines tokens with AND semantics. A qualifier with no value is
treated as a literal term so a partially typed query never throws. Matching
returns both note paths and cluster IDs, plus counts for the result panel.

## Explorer interaction

1. Render an always-visible search input above the hierarchy and UMAP canvas.
   Input changes are debounced (~75 ms), retain focus across renders when
   possible, and update only local view state.
2. Show a compact result summary/panel with matching note count, matching
   cluster count, active query, and an empty-state explanation. Note buttons
   and hierarchy nodes carry `is-match`/`is-dimmed` classes; ancestors stay
   visible so a match is never hidden by a collapsed branch.
3. Add combinable chips: All, Current cluster, Noise, Provisional, Manually
   adjusted, and Recently changed. Unsupported chips are visibly disabled and
   explain that the source field will arrive with manual-correction
   persistence. Recently changed uses persisted note mtime with a documented
   seven-day window.
4. Add Focus controls to cluster nodes. Focus state is a node ID (or root),
   with breadcrumbs, parent/root exit, and sibling next/previous navigation
   when a parent is available. `Escape` leaves focus; a root focus is a safe
   no-op. Focus scopes matching, counts, and note lists to the selected
   subtree while preserving ancestor breadcrumbs.
5. Keyboard behavior: `/` and Cmd/Ctrl+F focus the search field, `Escape`
   clears the query (or exits Focus when the query is already empty), `Enter`
   opens the first result, and Up/Down move the active result. Buttons retain
   normal Obsidian keyboard semantics and expose labels/pressed state.

## Performance and correctness

The index is built once per result/metadata revision and stores normalized
strings plus token sets; each query scans pre-tokenized fields and is debounced
at the UI boundary. No embedding, UMAP, HDBSCAN, or title-generation call is
made from search or focus handlers. A deterministic 10,000-document fixture
will assert query latency stays within a generous test budget, result order is
path-stable, and all required metadata fields participate when present.

Focused regression tests will cover parser qualifiers/phrases, filter
composition, ancestor retention, optional tags/aliases fallback, focus exit
and keyboard bindings, unsupported manual-adjustment state, and the 10k
metadata fixture. Existing visualization and incremental-refresh suites must
remain unchanged and green.

## Acceptance checklist

- [ ] Search bar is visible in every Explorer result state.
- [ ] Plain, phrase, `tag:`, `path:`, and `cluster:` queries are deterministic.
- [ ] Matching notes/clusters are highlighted; nonmatches are dimmed and
      hierarchy ancestors remain visible.
- [ ] Result summary and all six combinable filter chips are present; missing
      schema fields use an explicit fallback/disabled state.
- [ ] Focus provides subtree view, breadcrumbs, sibling navigation, and
      Escape/root exit.
- [ ] `/`, Cmd/Ctrl+F, Escape, Enter, and arrow navigation work without a
      clustering call.
- [ ] 10,000-metadata fixture/benchmark and focused tests pass.
- [ ] TypeScript, plugin build, full tests, and release/WASM checks are run;
      unavailable generated WASM tooling is recorded rather than fabricated.
