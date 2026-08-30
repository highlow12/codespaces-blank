# Wikipedia Soft Benchmark v1

The repository contains a promoted 720-row manifest of fixed `oldid`
revisions. The promotion was explicitly authorized for all automatically
checked candidates; it is not a claim that each row received independent
human review. Generated source, chunk, and package artifacts are ignored by
Git.

The package uses English Wikipedia text under CC BY-SA 4.0. See the package
attribution file for page/history URLs and modification notices.

Production token counts use `BAAI/bge-base-en-v1.5` at immutable revision
`a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`; the exact contract is in
`config.json`.

The candidate ledger may contain historical rejected rows in addition to the
720 approved release rows. Rejected rows retain reviewer, timestamp, and
reason metadata; they are audit history and are never copied to the release
manifest. Short cleaned bodies are recorded with the reason
`cleaned body cannot produce a 100-token chunk`.

The committed `validation_report.json` is a truthful pre-release baseline. A
successful `package` run writes the validated report into the generated
package; it does not alter this committed baseline report.

## Candidate review ledger

`candidate_manifest.jsonl` records the approved rows plus any retained
historical rejection records. It must contain exactly 720 approved rows with
36 discovery, 12 calibration, and 12 test rows per leaf. The release
`source_manifest.jsonl` contains only those 720 approved rows. The ledger
retains MediaWiki search evidence, automatic page checks, and review metadata
for both approved and rejected candidates.
